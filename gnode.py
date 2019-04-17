from __future__ import annotations

from collections import namedtuple
from copy import deepcopy, copy
from typing import Tuple, Iterable, Type

from vtypes import VType

attr_types = {
    'zero': 10,
    'one': 10,
    'notbit': 5,
    'neg': 4,
    'invdiv': 4
}

AttrGroup = namedtuple('attr_group', ('type', 'attrs'))


class GNode:
    a = ()
    val_args = ()

    def __init__(self, p):
        self.p = p
        self.op = None
        self.attr_stack = []
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

        return self.gen_op('and', self, v)

    def __or__(self, v):
        if self.notbot and v.notbit:
            return ~((~self) & (~v))
        if self.zero:
            return v
        if v.zero:
            return self

        return self.gen_op('or', self, v)

    def __xor__(self, v):
        if self.issame(v):
            return self.gen_zero()
        if self.notbit and v.notbit:
            return (~self) ^ (~v)
        if self.zero:
            return self
        if v.zero:
            return v

        return self.gen_op('xor', self, v)

    def shufl(self, v, c=None, r=None):
        n = 'shufl'
        if c:
            n += 'c' + c
        if r:
            n += 'r' + r

        self.gen_op(n, self, v)

    def shufh(self, v, c=None, r=None):
        n = 'shufh'
        if c:
            n += 'c' + c
        if r:
            n += 'r' + r

        return self.gen_op(n, self, v)

    def rotl(self, v, c=None, r=None):
        n = 'rotl'
        if c:
            n += 'c' + c
        if r:
            n += 'r' + r

        self.gen_op(n, self, v)

    def rotr(self, v, c=None, r=None):
        n = 'rotr'
        if c:
            n += 'c' + c
        if r:
            n += 'r' + r

        return self.gen_op(n, self, v)

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
    def __init__(self, p, n: str, a: Iterable[GNode]):
        super().__init__(p)
        self.a: Tuple[GNode] = tuple(a)
        self.op = p.find_op(n, a)


class StoreNode(GNode):
    def __init__(self, p, v: GNode, val):
        super().__init__(p)
        self.a: Tuple[GNode] = (v, )
        self.op = p.find_store_op(v.op.out_t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(super().key, self.val)


class ConstNode(GNode):
    def __init__(self, p, t: VType, val):
        super().__init__(p)
        self.op = p.find_const_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op.name, self.val)


class LoadNode(GNode):
    def __init__(self, p, t: VType, val):
        super().__init__(p)
        self.op = p.find_load_op(t)
        self.val = val

    @property
    def val_args(self):
        return self.val,

    @property
    def key(self) -> str:
        return '{}Z{}'.format(self.op.name, self.val)


class CvtNode(GNode):
    def __init__(self, p, v: GNode, t: VType):
        super().__init__(p)
        self.a = (v, )
        self.op = p.find_cvt_op(v, t)

