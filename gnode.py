from __future__ import annotations

from copy import deepcopy, copy
from typing import Tuple, Iterable, List, NamedTuple, Set, Optional

from utils import get_p2, str_list, flush_attrs
from vtypes import VType, OpDescr

attr_types = {
    'zero': 10,
    'one': 10,
    'notbit': 5,
    'neg': 4,
    'invdiv': 4
}

AttrGroup = NamedTuple('AttrGroup', (('type', int), ('attrs', Set[str])))
OpScope = NamedTuple('OpScope', (('start_pos', int), ('end_pos', int), ('exp_use', float)))


class GNode:
    a = ()
    val_args = ()
    var_name = None

    def __init__(self, p, scope_n: int):
        self.p = p
        self.scope_n: int = scope_n
        self.op: Optional[OpDescr] = None
        self.attr_stack: List[AttrGroup] = []
        self.num_attrs = set()

    @property
    def type(self) -> VType:
        return self.op.out_t

    @property
    def op_name(self):
        return self.op.name

    def flush_attr(self) -> GNode:
        new_node = self.copy()
        attr_stack = new_node.attr_stack
        new_node.attr_stack = ()
        for grp in attr_stack:
            for a in sorted(grp.attrs):
                new_node = getattr(new_node, 'apply_' + a)()
        return new_node

    def apply_notbit(self) -> GNode:
        return self.p.op('notbit', self)

    def apply_inv(self) -> GNode:
        return self.p.op('invdiv', self)

    def apply_neg(self) -> GNode:
        return self.p.op('neg', self)

    def __add__(self, v: GNode) -> GNode:
        if v.neg:
            return self - (-v)
        if self.neg:
            return v - (-self)
        if v.zero:
            return self
        if self.zero:
            return v
        return self.gen_op('add', self, v)

    def __sub__(self, v: GNode) -> GNode:
        if self.issame(v):
            return self.gen_zero()
        if self.neg:
            return -((-self) + v)
        if v.neg:
            return self + (-v)
        if v.zero:
            return self
        if self.zero:
            return -v
        return self.gen_op('sub', self, v)

    def __mul__(self, v: GNode) -> GNode:
        if self.neg:
            return -(v * (-self))
        # if self.get('invdiv'):
        #     return (v * self.setnot('invdiv')).setnot('invdiv')
        if self.zero or v.one:
            return self
        if self.one or v.zero:
            return v
        return self.gen_op('mul', self, v)

    def __truediv__(self, v: GNode) -> GNode:
        if self.issame(v):
            return self.gen_one()
        if self.zero:
            return self
        if v.one:
            return self
        if self.one:
            return v.setnot('invdiv')
        return self.gen_op('div', self, v)

    def __neg__(self) -> GNode:
        return self.setnot('neg')

    def __invert__(self) -> GNode:
        return self.setnot('notbit')

    def __and__(self, v: GNode) -> GNode:
        if self.notbit and v.notbit:
            return ~((~self) | (~v))
        if self.zero:
            return self.copy()
        if v.zero:
            return v.copy()

        return self.gen_op('and', self, v)

    def __or__(self, v: GNode) -> GNode:
        if self.notbot and v.notbit:
            return ~((~self) & (~v))
        if self.zero:
            return v.copy()
        if v.zero:
            return self.copy()

        return self.gen_op('or', self, v)

    def __xor__(self, v: GNode) -> GNode:
        if self.issame(v):
            return self.gen_zero()
        if self.notbit and v.notbit:
            return (~self) ^ (~v)
        if self.zero:
            return self.copy()
        if v.zero:
            return v.copy()

        return self.gen_op('xor', self, v)

    def nop(self):
        return self.gen_op('nop', self)

    def shuf(self, v: GNode, *dshifts) -> Tuple[GNode, GNode]:
        n = ''
        for i, s in enumerate(dshifts):
            if s:
                n += 'X{}S{}'.format(i, s)

        return self.gen_op('shufl' + n, self, v), \
               self.gen_op('shufh' + n, self, v)

    @staticmethod
    def shuf_dims(n: int, vecs: Iterable[GNode]) -> Tuple[GNode]:
        vecs: List[GNode] = list(vecs)
        if vecs:
            vdims = vecs[0].type.dims
            assert n < len(vdims)

            dim_sz = vdims[n]
            dim_shuf = tuple(int(i == n) for i in range(len(vdims)))
            assert len(vecs) == dim_sz

            for it_n in reversed(range(get_p2(dim_sz))):
                s = 2 ** it_n
                for i in range(0, dim_sz, s * 2):
                    for j in range(s):
                        vecs[i + j], vecs[i + j + s] = vecs[i + j].shuf(vecs[i + j + s], *dim_shuf)
        return tuple(vecs)

    def rotp(self, v: GNode, p: int, dn: int):
        dim_sz = v.type.dims[dn]
        assert p <= dim_sz
        if not p:
            return self.copy()
        if p == dim_sz:
            return v.copy()
        v_hlp = self.gen_op('mvhalf1h2l{}'.format(dn), self, v)
        if p == dim_sz // 2:
            return v_hlp
        if p < dim_sz // 2:
            return self.gen_op('rothalfd{}p{}'.format(dn, p), self, v_hlp)
        else:
            return self.gen_op('rothalfd{}p{}'.format(dn, p - dim_sz / 2), v_hlp, v)

    def setnot(self, a: str) -> GNode:
        new_node = self.copy()
        type = attr_types[a]

        if new_node.attr_stack and new_node.attr_stack[-1].type == type:
            attrs = new_node.attr_stack[-1].attrs
            if a in attrs:
                attrs.discard(a)
            else:
                attrs.add(a)
        else:
            new_node.attr_stack.append(AttrGroup(type, {a}))

        while new_node.attr_stack and not new_node.attr_stack[-1].attrs:
            new_node.attr_stack.pop()

        return new_node

    def __getattr__(self, a: str) -> bool:
        if a.startswith('_'):
            return super().__getattribute__(a)
        if not self.attr_stack:
            return a in self.num_attrs
        return a in self.attr_stack[-1].attrs

    def gen_op(self, n: str, *a: GNode) -> GNode:
        return self.p.op(n, *a)

    def gen_zero(self) -> GNode:
        ret = self.p.zero(self.type)
        ret.attr_stack.add('zero')
        return ret

    def gen_one(self) -> GNode:
        ret = self.p.const(self.type, 1.0)
        ret.attr_stack.add('one')
        return ret

    def store(self, val):
        self.p.store(self, val)

    @property
    def orig(self):
        return self.__dict__.get('_orig', id(self))

    def copy(self) -> GNode:
        r = copy(self)
        r._orig = self.orig
        r.attr_stack = deepcopy(self.attr_stack)
        r.num_attrs = deepcopy(self.num_attrs)
        return r

    def reset_orig(self):
        r = self.copy()
        r._orig = id(r)
        self.p.add_alias(self, r)
        return r

    def issame(self, v: GNode) -> GNode:
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
                self.op_name,
                'X'.join(str(o.orig) for o in self.a)
            )
        else:
            return '{}Y{}'.format(
                self.op_name,
                'X'.join(sorted(str(o.orig) for o in self.a))
            )

    def print_op(self, mapper):
        op_call = '{}({});'.format(
            self.op_name,
            ', '.join(
                str_list(self.val_args) + tuple(map(mapper, self.a))
            )
        )
        if self.type:
            print('{} {} = {}'.format(self.type, mapper(self), op_call))
        else:
            print(op_call)


class OpNode(GNode):
    def __init__(self, p, n: str, a: Iterable[GNode]):
        a = flush_attrs(a)
        super().__init__(p, p.get_scope_n(*a))
        self.a: Tuple[GNode] = tuple(a)
        self.op = p.find_op(n, a)


class StoreNode(GNode):
    def __init__(self, p, v: GNode, val):
        v = v.flush_attr()
        super().__init__(p, p.get_scope_n(v))
        self.a: Tuple[GNode] = (v,)
        self.op = p.find_store_op(v.type)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(super().key, self.val)


class VarNode(GNode):
    def __init__(self, p, t: VType, var_name: str):
        super().__init__(p, p.get_scope_n())
        self.t = t
        self.var_name = var_name

    @property
    def type(self):
        return self.t

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.var_name, self.t)

    def print_op(self, mapper):
        pass


class ConstNode(GNode):
    def __init__(self, p, t: VType, val):
        super().__init__(p, p.get_scope_n())
        self.op = p.find_const_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op_name, self.val)


class LoadNode(GNode):
    def __init__(self, p, t: VType, val):
        super().__init__(p, p.get_scope_n())
        self.op = p.find_load_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op_name, self.val)


class CvtNode(GNode):
    def __init__(self, p, v: GNode, t: VType):
        v = v.flush_attr()
        super().__init__(p, p.get_scope_n(v))
        self.a = (v,)
        self.op = p.find_cvt_op(v, t)


class CodeNode(GNode):
    def __init__(self, p, code, a: Iterable[GNode]):
        a = flush_attrs(a)
        super().__init__(p, p.get_scope_n(*a))
        self.a: Tuple[GNode] = tuple(a)
        self.code = code

    @property
    def type(self):
        return None

    @property
    def val_args(self):
        return self.code,

    @property  # do not remove duplicates for code
    def key(self) -> str:
        return str(self.orig)

    def print_op(self, mapper):
        print(
            self.code.format(*map(mapper, self.a))
        )
