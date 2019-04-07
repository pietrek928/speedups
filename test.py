from collections import namedtuple, defaultdict
from copy import deepcopy, copy
from typing import Tuple, Iterable, List

from optim import _proc, _prog, test


class ProcCtx:

    def __init__(self, mem_levels, ops):
        self.ops = {}
        ports = set(sum(
            (o[3] for o in ops),
            tuple(m[2] for m in mem_levels)
        ))
        p = _proc(len(ports))
        port2n = dict(zip(*reversed(list(zip(*enumerate(ports))))))
        for n, size, port, load_time in mem_levels:
            p.new_mem_level(
                size, port2n[port], load_time
            )
        for n, out_t, len_t, ports, ordered in ops:
            op_id = p.new_op(
                len_t,
                [port2n[p] for p in ports]
            )
            self.ops[n] = OpDescr(n, op_id, ordered, out_t)
        self.op_g = {}
        self._proc = p
        dict()

    def find_op(self, n, a):
        t = tuple(v.op.out_t for v in a)
        try:
            op_n = '{}Y{}'.format(n, 'X'.join(t))
            return self.ops[op_n]
        except KeyError:
            op_n = '{}Y{}'.format(n, 'X'.join(sorted(t)))
            op = self.ops[op_n]
            if not op.ordered:
                return op
            raise ValueError('Wrong argument order')

    def find_const_op(self, t):
        op_n = 'constY{}'.format(t)
        return self.ops[op_n]

    def find_load_op(self, t):
        op_n = 'loadY{}'.format(t)
        return self.ops[op_n]

    def find_cvt_op(self, v, t):
        op_n = 'cvtY{}X{}'.format(v.op.out_t, t)
        return self.ops[op_n]

    def find_store_op(self, t):
        op_n = 'storeY{}'.format(t)
        return self.ops[op_n]

    def _add_n(self, v):
        k = v.key
        if k in self.op_g:
            return self.op_g[k].copy()
        else:
            self.op_g[k] = v
            return v

    def op(self, n, *a):
        return self._add_n(
            OpNode(self, n, a)
        )

    def cvt(self, v, t):
        return self._add_n(
            CvtNode(self, v, t)
        )

    def load(self, t, val):
        return self._add_n(
            LoadNode(self, t, val)
        )

    def store(self, v, val):
        return self._add_n(
            StoreNode(self, v, val)
        )

    def const(self, t, v):
        # v = self._format_const(t, v)
        return self._add_n(
            ConstNode(self, t, v)
        )

    def zero(self, t):
        return self._add_n(
            OpNode(self, 'zeroY{}'.format(t), ())
        )

    def select_used(self):
        stack = {}
        used = set()

        for v in self.op_g.values():
            if v.op.out_t is None:
                stack[v.orig] = v

        used_items = []
        while stack:
            o, v = stack.popitem()
            used.add(o)

            used_items.append(v)
            for nv in v.a:
                if nv.orig not in used:
                    stack[nv.orig] = nv

        return used_items

    def used_ordered(self):
        g_rev = defaultdict(lambda: [])
        cnt = {}

        used = self.select_used()
        for v in used:
            cnt[v.orig] = len(v.a)
            for nv in v.a:
                g_rev[nv.orig].append(v)

        stack = []
        for v in used:
            if not cnt[v.orig]:
                stack.append(v)

        n = 1
        ordered = []
        while stack:
            v = stack.pop()
            ordered.append(v)
            n += 1
            for nv in g_rev[v.orig]:
                cnt[nv.orig] -= 1
                if not cnt[nv.orig]:
                    stack.append(nv)

        return ProgGraph(self._proc, ordered)

    def print_graph(self):
        g_rev = defaultdict(lambda: [])
        cnt = {}

        used = self.select_used()
        for v in used:
            cnt[v.orig] = len(v.a)
            for nv in v.a:
                g_rev[nv.orig].append(v)

        stack = []
        for v in used:
            if not cnt[v.orig]:
                stack.append(v)

        nums = {}
        n = 1
        while stack:
            v = stack.pop()
            n += 1
            nums[v.orig] = n
            for nv in g_rev[v.orig]:
                cnt[nv.orig] -= 1
                if not cnt[nv.orig]:
                    stack.append(nv)
            print('{}: {} {}'.format(
                n, v.key, ' '.join(str(nums[nv.orig]) for nv in v.a)))


OpDescr = namedtuple('op_descr', ('name', 'op_id', 'ordered', 'out_t'))
AttrGroup = namedtuple('attr_group', ('type', 'attrs'))


attr_types = {
    'zero': 10,
    'one': 10,
    'notbit': 5,
    'neg': 4,
    'invdiv': 4
}


class GNode:
    a = ()
    val_args = ()

    def __init__(self, p: ProcCtx):
        self.p: ProcCtx = p
        self.op: OpDescr = None
        self.attr_stack: List[AttrGroup] = []
        self.num_attrs = set()

    def flush_attr(self):
        self = self.copy()
        for grp in self.attr_stack:
            for a in sorted(grp.attrs):
                self = getattr(self, 'apply_' + a)()
        return self

    def apply_notbit(self):
        return self.p.op('notbit', self)

    def apply_inv(self):
        return self.p.op('invdiv', self)

    def apply_neg(self):
        return self.p.op('neg', self)

    def __add__(self, v):
        if v.neg:
            return self-(-v)
        if self.neg:
            return v-(-self)
        if v.zero:
            return self
        if self.zero:
            return v
        return self.gen_op('add', self, v)

    def __sub__(self, v):
        if self.issame(v):
            return self.gen_zero()
        if self.neg:
            return -((-self)+v)
        if v.neg:
            return self+(-v)
        if v.zero:
            return self
        if self.zero:
            return -v
        return self.gen_op('sub', self, v)

    def __mul__(self, v):
        if self.neg:
            return -(v * (-self))
        # if self.get('invdiv'):
        #     return (v * self.setnot('invdiv')).setnot('invdiv')
        if self.zero or v.one:
            return self
        if self.one or v.zero:
            return v
        return self.gen_op('mul', self, v)

    def __truediv__(self, v):
        if self.issame(v):
            return self.gen_one()
        if self.zero:
            return self
        if v.one:
            return self
        if self.one:
            return v.setnot('invdiv')
        return self.gen_op('div', self, v)

    def __neg__(self):
        return self.setnot('neg')

    def __invert__(self):
        return self.setnot('notbit')

    def __and__(self, v):
        if self.notbit and v.notbit:
            return ~((~self) | (~v))
        if self.zero:
            return self
        if v.zero:
            return v

        self.gen_op('and', self, v)

    def __or__(self, v):
        if self.notbot and v.notbit:
            return ~((~self) & (~v))
        if self.zero:
            return v
        if v.zero:
            return self

        self.gen_op('or', self, v)

    def __xor__(self, v):
        if self.issame(v):
            return self.gen_zero()
        if self.notbit and v.notbit:
            return (~self) ^ (~v)
        if self.zero:
            return self
        if v.zero:
            return v

        self.gen_op('xor', self, v)

    def setnot(self, a):
        self = self.copy()
        type = attr_types[a]

        if self.attr_stack and self.attr_stack[-1].type == type:
            attrs = self.attr_stack[-1].attrs
            if a in attrs:
                attrs.discard(a)
            else:
                attrs.add(a)
        else:
            self.attr_stack.append(AttrGroup(type, {a}))

        while self.attr_stack and not self.attr_stack[-1].attrs:
            self.attr_stack.pop()

        return self

    def __getattr__(self, a):
        if a.startswith('_'):
            return super().__getattribute__(a)
        if not self.attr_stack:
            return a in self.num_attrs
        return a in self.attr_stack[-1].attrs

    def gen_op(self, n, *a):
        a = [v.flush_attr() for v in a]
        return self.p.op(n, *a)

    def gen_zero(self):
        ret = self.p.zero(self.op.out_t)
        ret.attr_stack.add('zero')
        return ret

    def gen_one(self):
        ret = self.p.const(self.op.out_t, 1.0)
        ret.attr_stack.add('one')
        return ret

    @property
    def orig(self):
        return self.__dict__.get('_orig', id(self))

    def copy(self):
        r = copy(self)
        r._orig = self.orig
        r.attr_stack = deepcopy(self.attr_stack)
        r.num_attrs = deepcopy(self.num_attrs)
        return r

    def issame(self, v):
        return (
            self.orig == v.orig
            and self.key == v.key
            and self.num_attrs == v.num_attrs
            and self.attr_stack == v.attr_stack
        )

    @property
    def key(self) -> str:
        if self.op.ordered:
            return '{}Y{}'.format(
                self.op.name,
                'X'.join(str(o.orig) for o in self.a)
            )
        else:
            return '{}Y{}'.format(
                self.op.name,
                'X'.join(sorted(str(o.orig) for o in self.a))
            )

    def print_op(self, nums):
        op_call = '{}({});'.format(
            self.op.name,
            ', '.join([
                str(val) for val in self.val_args
            ] + [
                'v{}'.format(nums[v.orig]) for v in self.a
            ])
        )
        if self.op.out_t:
            print('{} v{} = {}'.format(self.op.out_t, nums[self.orig], op_call))
        else:
            print(op_call)


class OpNode(GNode):
    def __init__(self, p: ProcCtx, n: str, a: Iterable[GNode]):
        super().__init__(p)
        self.a: Tuple[GNode] = tuple(a)
        self.op: OpDescr = p.find_op(n, a)


class StoreNode(GNode):
    def __init__(self, p: ProcCtx, v: GNode, val):
        super().__init__(p)
        self.a: Tuple[GNode] = (v, )
        self.op: OpDescr = p.find_store_op(v.op.out_t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(super().key, self.val)


class ConstNode(GNode):
    def __init__(self, p: ProcCtx, t: str, val):
        super().__init__(p)
        self.op: OpDescr = p.find_const_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op.name, self.val)


class LoadNode(GNode):
    def __init__(self, p: ProcCtx, t: str, val):
        super().__init__(p)
        self.op: OpDescr = p.find_load_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op.name, self.val)


class CvtNode(GNode):
    def __init__(self, p: ProcCtx, v: GNode, t: str):
        super().__init__(p)
        self.a = (v, )
        self.op: OpDescr = p.find_cvt_op(v, t)


class ProgGraph:
    def __init__(self, p: ProcCtx, op_l: List[GNode]):
        self.p: ProcCtx = p
        self.op_l: List[GNode] = op_l
        self._prog = _prog(p, self.op_nums, self._simple_graph())

    @property
    def op_nums(self):
        return tuple(
            v.op.op_id for v in self.op_l
        )

    def _simple_graph(self, ord=None):
        if ord is None:
            ord = range(len(self.op_l))

        G = []
        nums = {}
        for i in ord:
            v = self.op_l[i]
            nums[v.orig] = i
            G.append(tuple(
                nums[vn.orig] for vn in v.a
            ))

        return G

    def gen_code(self, ord=None):
        if ord is None:
            ord = range(len(self.op_l))

        nums = {}
        for i in ord:
            v = self.op_l[i]
            nums[v.orig] = i
            v.print_op(nums)


mem_levels = (
    ('regs', 16, 0, 0.0),
    ('L1', 3200, 1, 7.0)
)

ops = (
    ('loadXv4', 'v4', 7.0, (1, 2), True),
    ('storeYv4', None, 7.0, (6, 1), True),
    ('storeYfloat', None, 7.0, (6, 1), True),
    ('loadYv4', 'v4', 7.0, (3, ), True),
    ('addYv4Xv4', 'v4', 3.5, (3, 4), True),
    ('subYv4Xv4', 'v4', 3.0, (3, 4), True),
    ('mulYv4Xv4', 'v4', 5.5, (5, ), True),
    ('negYv4', 'v4', 5.0, (5, ), True),
    ('notbitYv4', 'v4', 5.0, (5, ), True),
    ('cvtYv4Xfloat', 'float', 5.0, (7, ), True)
)

p = ProcCtx(mem_levels=mem_levels, ops=ops)

a = p.load('v4', '&a')
b = p.load('v4', '&b')
c = p.load('v4', '&c')
d = p.load('v4', '&d')
b += a
d += c
a *= b
c *= d
b += a
d += c
c = a * (-~-b)
c = p.cvt(c, 'float')
p.store(a, '&a')
p.store(b, '&b')
p.store(c, '&c')
p.store(d, '&d')
# p.print_graph()
p.used_ordered().gen_code()

test(p.used_ordered()._prog)
