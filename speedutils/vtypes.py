from typing import Iterable, NamedTuple, TYPE_CHECKING

from .graphval import GraphVal, VType
from .proc_ctx import graph_ctx

if TYPE_CHECKING:
    from .flow import NodeMapper


class CodeVal(GraphVal):
    has_output = False

    def __init__(self, code, a: Iterable[GraphVal]):
        super().__init__(
            a=a
        )
        self.code = code
        self.key = str(self.orig)  # do not remove duplicates for code
        self.val_args = (self.code,)

    def print_op(self, mapper: 'NodeMapper'):
        print(
            self.code.format(*map(mapper.get, self.a))
        )

    @classmethod
    def from_code(cls, code: str, *a: GraphVal) -> GraphVal:
        v = cls(code=code, a=a)
        return v.p.add_node(v)

    @classmethod
    def stationary_code(cls, code: str, *a: GraphVal) -> GraphVal:
        graph_ctx.new_scope()
        v = cls(code=code, a=a)
        v.scope_n = graph_ctx.get_scope_n()
        r = v.p.add_node(v)
        graph_ctx.new_scope()
        return r


# class VType:
#     name: str = '?'
#     shape: Tuple[int] = ()
#
#     def __call__(self, val):
#         return func_ctx.v(val, self)
#
#     def format(self, v):
#         return v
#
#     def trunc(self, v):
#         return str(self.format(v))
#
#     def v(self, val):  # TODO: remove
#         return func_ctx.v(val, self)
#
#     def zero(self):
#         return graph_ctx.zero(self)
#
#     def one(self):
#         return graph_ctx.one(self)
#
#     def load(self, arr: GraphVal, idx: GraphVal):
#         return graph_ctx.load(self, arr, idx)
#
#     def var(self, name: str, const=False, default=None):
#         return func_ctx.get_var(name, self, const=const, default=default)
#
#     def __eq__(self, v: 'VType'):
#         return self.name == v.name
#
#     def __str__(self):
#         return self.name
#
#     def __repr__(self):
#         return str(self)


# class Ptr(GraphVal):
#     def __init__(self, t: VType):
#         self._t: VType = t
#
#     def __str__(self):
#         return f'{self._t}*'
#
#     @property
#     def dims(self):
#         raise ValueError('`ptr` has no dimensions')
#
#     @property
#     def name(self):
#         return f'ptrX{self._t.name}'


OpDescr = NamedTuple('op_descr', (('name', str), ('op_id', int), ('ordered', int), ('out_t', VType)))


class bool_(GraphVal):
    type_name = 'bool'

    @classmethod
    def from_const(cls, val):
        super().from_const(bool(val))


class int32_(GraphVal):
    type_name = 'int32'

    @classmethod
    def from_const(cls, val):  # TODO: check range
        super().from_const(int(val))


class float_(GraphVal):
    type_name = 'float'

    @classmethod
    def from_const(cls, val):  # TODO: check nan ?
        super().from_const(float(val))


class v4f(GraphVal):
    type_name = 'v4f'
    shape = (4,)

    @classmethod
    def from_const(cls, val):
        items = tuple(
            map(float, val)
        )
        super().from_const(items)


# class v4x2f_(VType):
#     name = 'v4x2f'
#     shape = (4, 2)


def _trunc_cfg(obj) -> str:
    if isinstance(obj, dict):
        return _trunc_cfg(
            tuple(obj.items())
        )
    elif isinstance(obj, (list, tuple)):
        return 'I'.join(
            sorted(map(_trunc_cfg, obj))
        )
    else:
        return str(obj)


class Tcfg(GraphVal):
    name = 'cfg'

    @property
    def dims(self):
        raise ValueError('cfg has no dimensions')

    # def format(self, v) -> str:
    #     return v

    def trunc(self) -> str:
        return _trunc_cfg(self.val)

# v4f = v4f_()
# v4x2f = v4x2f_()
# Tcfg = Tcfg_()
