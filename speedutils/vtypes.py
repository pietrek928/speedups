from typing import NamedTuple, TYPE_CHECKING, Tuple

from .proc_ctx import func_ctx, graph_ctx

if TYPE_CHECKING:
    from .gnode import GNode


class VType:
    name: str = '?'
    shape: Tuple[int] = ()

    def __call__(self, val):
        return func_ctx.v(val, self)

    def format(self, v):
        return v

    def trunc(self, v):
        return str(self.format(v))

    def v(self, val):  # TODO: remove
        return func_ctx.v(val, self)

    def zero(self):
        return graph_ctx.zero(self)

    def one(self):
        return graph_ctx.one(self)

    def load(self, arr: GNode, idx: GNode):
        return graph_ctx.load(self, arr, idx)

    def var(self, name: str, const=False, default=None):
        return func_ctx.get_var(name, self, const=const, default=default)

    def __eq__(self, v: 'VType'):
        return self.name == v.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class Ptr(VType):
    def __init__(self, t: VType):
        self._t: VType = t

    def __str__(self):
        return f'{self._t}*'

    @property
    def dims(self):
        raise ValueError('`ptr` has no dimensions')

    @property
    def name(self):
        return f'ptrX{self._t.name}'


OpDescr = NamedTuple('op_descr', (('name', str), ('op_id', int), ('ordered', int), ('out_t', VType)))


class bool__(VType):
    name = 'bool'

    def format(self, v) -> bool:
        return bool(v)


class int32__(VType):
    name = 'int32'

    def format(self, v) -> int:  # TODO: check range
        return int(v)


class float__(VType):
    name = 'float'

    def format(self, v) -> float:
        return float(v)


class v4f_(VType):
    name = 'v4f'
    shape = (4,)


class v4x2f_(VType):
    name = 'v4x2f'
    shape = (4, 2)


def _format_cfg(obj):
    if isinstance(obj, dict):
        return _format_cfg(
            tuple(obj.items())
        )
    elif isinstance(obj, (list, tuple)):
        return 'I'.join(
            sorted(map(_format_cfg, obj))
        )
    else:
        return str(obj)


class Tcfg_(VType):
    name = 'cfg'

    @property
    def dims(self):
        raise ValueError('cfg has no dimensions')

    def format(self, v) -> str:
        return v

    def trunc(self, v):
        return _format_cfg(v)


bool_ = bool__()
int32_ = int32__()
float_ = float__()
v4f = v4f_()
v4x2f = v4x2f_()
Tcfg = Tcfg_()
