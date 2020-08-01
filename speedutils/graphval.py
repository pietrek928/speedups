from __future__ import annotations

from copy import copy, deepcopy
from typing import Iterable, List, NamedTuple, Optional, Set, TYPE_CHECKING, Tuple, Type

from .proc_ctx import graph_ctx
from .utils import str_list

if TYPE_CHECKING:
    from .vtypes import OpDescr
    from .flow import FlowGraph, NodeMapper
    from .proc_descr import Op

_orig_id = 1234


def _next_orig() -> int:
    global _orig_id
    _orig_id += 1
    return _orig_id


attr_types = {
    'zero': 10,
    'one': 10,
    'notbit': 5,
    'neg': 4,
    'invdiv': 4
}

AttrGroup = NamedTuple('AttrGroup', (('type', int), ('attrs', Set[str])))
OpScope = NamedTuple('OpScope', (('start_pos', int), ('end_pos', int), ('exp_use', float)))
VType = Type['GraphVal']


class GraphVal:
    op_name: str = ''
    val_args = ()
    var_name: Optional[str] = None
    val = ''
    const: bool = False
    type_name: str = '?'
    comment = None
    name_prefix = None
    is_rendered = True

    dims = (1,)

    def __init__(
            self,
            a: Iterable['GraphVal', ...] = (), op: Optional['OpDescr'] = None,
            code=None, remove_duplicates=True
    ):
        self.p: 'FlowGraph' = graph_ctx.value
        self.scope_n: int = self.p.get_scope_n(*a)
        self.a = tuple(
            v.flush_attr() for v in a
        )

        self.attr_stack: List[AttrGroup] = []
        self.num_attrs = set()
        self.orig = _next_orig()

        self.op: Op = op
        self.code = code
        self.op_name = self.code
        if self.op is not None:
            self.op_name = self.op_name or self.op.name
        else:
            remove_duplicates = False

        if remove_duplicates:
            self.key = self._gen_key()
        else:
            str(self.orig)

    def trunc(self) -> str:
        if self.const:
            return str(self.val)
        return str(self.var_name or self.orig or '')

    def istype(self, *t: VType):
        return isinstance(self, t)

    def set_comment(self, comment):
        self.comment = comment
        return self

    def set_prefix(self, name_prefix):
        self.name_prefix = name_prefix
        return self

    @property
    def has_output(self):
        return self.op is not None and self.op.ret_t is not None

    def flush_attr(self) -> GraphVal:
        new_node = self.copy()
        attr_stack = new_node.attr_stack
        new_node.attr_stack = ()
        for grp in attr_stack:
            for a in sorted(grp.attrs):
                new_node = getattr(new_node, 'apply_' + a)()
        return new_node

    def apply_notbit(self) -> GraphVal:
        return self.from_op('notbit', self)

    def apply_inv(self) -> GraphVal:
        return self.from_op('invdiv', self)

    def apply_neg(self) -> GraphVal:
        return self.from_op('neg', self)

    def __add__(self, v: GraphVal) -> GraphVal:
        if v.neg:
            return self - (-v)
        if self.neg:
            return v - (-self)
        if v.zero:
            return self
        if self.zero:
            return v
        return self.from_op('add', self, v)

    def __sub__(self, v: GraphVal) -> GraphVal:
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
        return self.from_op('sub', self, v)

    def __mul__(self, v: GraphVal) -> GraphVal:
        if self.neg:
            return -(v * (-self))
        # if self.get('invdiv'):
        #     return (v * self.setnot('invdiv')).setnot('invdiv')
        if self.zero or v.one:
            return self
        if self.one or v.zero:
            return v
        return self.from_op('mul', self, v)

    def __truediv__(self, v: GraphVal) -> GraphVal:
        if self.issame(v):
            return self.gen_one()
        if self.zero:
            return self
        if v.one:
            return self
        if self.one:
            return v.setnot('invdiv')
        return self.from_op('div', self, v)

    def __neg__(self) -> GraphVal:
        return self.setnot('neg')

    def __invert__(self) -> GraphVal:
        return self.setnot('notbit')

    def __and__(self, v: GraphVal) -> GraphVal:
        if self.notbit and v.notbit:
            return ~((~self) | (~v))
        if self.zero:
            return self.copy()
        if v.zero:
            return v.copy()

        return self.from_op('and', self, v)

    def __or__(self, v: GraphVal) -> GraphVal:
        if self.notbit and v.notbit:
            return ~((~self) & (~v))
        if self.zero:
            return v.copy()
        if v.zero:
            return self.copy()

        return self.from_op('or', self, v)

    def __xor__(self, v: GraphVal) -> GraphVal:
        if self.issame(v):
            return self.gen_zero()
        if self.notbit and v.notbit:
            return (~self) ^ (~v)
        if self.zero:
            return self.copy()
        if v.zero:
            return v.copy()

        return self.from_op('xor', self, v)

    def nop(self):
        return self.from_op('nop', self)

    def shuf(self, v: GraphVal, *dshifts) -> Tuple[GraphVal, GraphVal]:
        n = ''
        for i, s in enumerate(dshifts):
            if s:
                n += f'X{i}S{s}'

        return self.from_op('shufl' + n, self, v), \
               self.from_op('shufh' + n, self, v)

    def rotp(self, v: GraphVal, p: int, dn: int):
        dim_sz = v.type.shape[dn]
        assert p <= dim_sz
        if not p:
            return self.copy()
        if p == dim_sz:
            return v.copy()
        v_hlp = self.from_op('mvhalf1h2l{}'.format(dn), self, v)
        if p == dim_sz // 2:
            return v_hlp
        if p < dim_sz // 2:
            return self.from_op('rothalfd{}p{}'.format(dn, p), self, v_hlp)
        else:
            return self.from_op('rothalfd{}p{}'.format(dn, p - dim_sz / 2), v_hlp, v)

    def setnot(self, a: str) -> GraphVal:
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

    # def gen_op(self, n: str, *a: GraphVal) -> GraphVal:
    #     return self.p.op(n, *a)

    # def gen_zero(self) -> GraphVal:
    #     return self.p.zero(self.type)
    #
    # def gen_one(self) -> GraphVal:
    #     return self.p.one(self.type)

    @classmethod
    def gen_zero(cls) -> GraphVal:
        v = cls.from_spec_op('zero')
        v.num_attrs.add('zero')
        return v.p.add_node(v)

    @classmethod
    def gen_one(cls) -> GraphVal:
        v = cls.from_const(1.0)
        v.num_attrs.add('one')
        return v.p.add_node(v)

    # def store(self, arr: GraphNode, val):
    #     self.p.store(self, arr, val)

    def copy(self) -> GraphVal:
        r = copy(self)
        r.attr_stack = deepcopy(self.attr_stack)
        r.num_attrs = deepcopy(self.num_attrs)
        return r

    def reset_orig(self):
        r = self.copy()
        r.orig = _next_orig()
        self.p.add_alias(self, r)
        return r

    def bind_scope(self) -> GraphVal:
        v = self.reset_orig()
        v.scope_n = graph_ctx.get_scope_n()
        return v  # FIXME: add to graph ??????

    def issame(self, v: GraphVal) -> bool:
        return (
                self.orig == v.orig
                and self.key == v.key
                and self.num_attrs == v.num_attrs
                and self.attr_stack == v.attr_stack
        )

    def _gen_key(self) -> str:
        if self.op.args_ordered:
            args_str = 'X'.join(str(o.orig) for o in self.a)
        else:
            args_str = 'X'.join(sorted(str(o.orig) for o in self.a))
        return f'{self.op_name}Y{args_str}'

    def render_op(self, mapper: NodeMapper) -> Iterable[str]:
        if not self.is_rendered:
            return

        arg_list = str_list(self.val_args) + tuple(map(mapper.get, self.a))
        if self.has_output:
            prefix = [f'{self.type_name} {mapper.get(self)} =']
        else:
            prefix = []
        if self.code:
            text = ' '.join(prefix + [
                self.code.format(*arg_list)
            ])
        else:
            # op_args = ', '.join(arg_list)
            # op_call = f'{self.op_name}({op_args});'
            text = ' '.join(prefix + [
                self.op.format_expr(arg_list)
            ])
        if self.comment is not None:  # TODO: multiline comment ?
            text += ' // ' + self.comment.replace('\n', ' ')
        yield text

    @classmethod
    def from_op(cls, n: str, *a: GraphVal):
        op = graph_ctx.find_op(n, a)
        v = op.ret_t(
            a=a,
            op=op
        )
        return v.p.add_node(v)

    @classmethod
    def from_spec_op(cls, n: str, *a: GraphVal):
        op = graph_ctx.find_spec_op(n, cls)
        v = op.ret_t(
            a=a,
            op=op
        )
        return v.p.add_node(v)

    @classmethod
    def from_expr(cls, code, *a: GraphVal, op=None):
        t: VType = cls
        if op is not None:
            t = op.ret_t or t

        v = t(
            a=a,
            op=op,
            code=code
        )
        return v.p.add_node(v)

    @classmethod
    def stationary_code(cls, code: str, op=None, *a: GraphVal) -> GraphVal:
        t: VType = cls
        if op is not None:
            t = op.ret_t or t

        graph_ctx.new_scope()
        v = t(
            a=a,
            op=op,
            code=code,
            remove_duplicates=False
        )
        v.scope_n = graph_ctx.get_scope_n()
        r = v.p.add_node(v)
        graph_ctx.new_scope()
        return r

    @classmethod
    def var(cls, var_name: str, start_scope=True):
        v = cls(
            op=graph_ctx.find_spec_op('load', cls)
        )

        v.key = f'{var_name}Z{cls.__name__}'
        v.var_name = var_name
        v.is_rendered = False
        if start_scope:
            v.scope_n = 0

        return v.p.add_node(v)

    @classmethod
    def load(cls, arr: GraphVal, idx: GraphVal):
        v = cls(
            a=(arr, idx),
            op=graph_ctx.find_load_op(cls)
        )
        return v.p.add_node(v)

    def store(self, arr: GraphVal, idx: GraphVal):
        v = type(self)(
            a=(self, arr, idx),
            op=graph_ctx.find_store_op(type(self))
        )
        return v.p.add_node(v)

    @classmethod
    def from_const(cls, val):
        v = cls(
            op=graph_ctx.find_const_op(cls)
        )

        v.val = val
        v.key = f'{v.op_name}Z{v.val}'
        v.val_args = (v.val,)
        v.const = True

        return v.p.add_node(v)

    def cvt(self, t: VType):
        if t is type(self):
            return self
        v = t(
            a=(self,),
            op=graph_ctx.find_cvt_op(self, t)
        )
        return v.p.add_node(v)

    def sep(self):
        v = type(self)(
            a=(self,),
            remove_duplicates=False
        )

        v.scope_n = v.p.get_scope_n()  # put separator in current scope
        return v.p.add_node(v)

# class OpNode(GraphNode):
#     def __init__(self, p: 'FlowGraph', n: str, a: Iterable[GraphNode]):
#         a = flush_attrs(a)
#         super().__init__(p, p.get_scope_n(*a))
#         self.a: Tuple[GraphNode] = tuple(a)
#         self.op = p.find_op(n, a)


# class StoreNode(GraphNode):
#     def __init__(self, p: 'FlowGraph', v: GraphNode, arr: GraphNode, idx: GraphNode):  # TODO: None node ?
#         v = v.flush_attr()
#         arr = arr.flush_attr()
#         idx = idx.flush_attr()
#         super().__init__(p, p.get_scope_n(v, arr, idx))
#         self.a = (v, arr, idx)
#         self.op = p.find_store_op(v.type)


# class LoadNode(GraphNode):
#     def __init__(self, p: 'FlowGraph', t: VType, arr: GraphNode, idx: GraphNode):
#         arr = arr.flush_attr()
#         idx = idx.flush_attr()
#         super().__init__(p, p.get_scope_n())
#         self.a = (arr, idx)
#         self.op = p.find_load_op(t)


# class VarNode(GraphNode):
#     def __init__(self, p: 'FlowGraph', t: VType, var_name: str):
#         super().__init__(p, p.get_scope_n())
#         self.t = t
#         self.var_name = var_name
#
#     @property
#     def type(self):
#         return self.t
#
#     @property
#     def key(self) -> str:
#         return '{}Z{}'.format(self.var_name, self.t)
#
#     def print_op(self, mapper):
#         pass


# class ConstNode(GraphVal):
#     const = True
#
#     def __init__(self, p: 'FlowGraph', t: VType, val):
#         super().__init__(p, p.get_scope_n())
#         self.op = p.find_const_op(t)
#         self.val = val
#
#     @property
#     def val_args(self):
#         return self.val,
#
#     @property
#     def key(self) -> str:
#         return f'{self.op_name}Z{self.val}'


# class CvtNode(GraphVal):
#     def __init__(self, p: 'FlowGraph', v: GraphVal, t: VType):
#         v = v.flush_attr()
#         super().__init__(p, p.get_scope_n(v))
#         self.a = (v,)
#         self.op = p.find_cvt_op(v, t)

# class SepNode(GraphNode):
#     op_name = ''
#
#     def __init__(self, p: 'FlowGraph', v: GraphNode):
#         v = v.flush_attr()
#         super().__init__(p, p.get_scope_n())
#         self.a: Tuple[GraphNode] = (v,)
#
#     @property
#     def type(self):
#         return self.a[0].type
#
#     @property  # do not remove duplicates
#     def key(self) -> str:
#         return str(self.orig)
