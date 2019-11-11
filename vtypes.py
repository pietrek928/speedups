from typing import Tuple, NamedTuple

from proc_ctx import graph_ctx, func_ctx


class VType:
    name: str = '?'
    dims: Tuple[int] = ()

    def format(self, v):
        return str(v)

    def v(self, val):
        return func_ctx.v(val, self)

    def zero(self):
        return graph_ctx.zero(self)

    def load(self, val):
        return graph_ctx.load(self, val)

    def var(self, name, const=False, default=None):
        return func_ctx.get_var(name, self, const=const, default=default)

    def __eq__(self, v):
        return self.name == v.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class ptr(VType):
    name = 'ptr'

    def __init__(self, t: VType):
        self._t: VType = t

    def __str__(self):
        return '{}*'.format(self._t)


OpDescr = NamedTuple('op_descr', (('name', str), ('op_id', int), ('ordered', int), ('out_t', VType)))


class bool__(VType):
    name = 'bool'

    def format(self, v):
        return bool(v)


class int32__(VType):
    name = 'int32'

    def format(self, v):  # TODO: check range
        return int(v)


class float__(VType):
    name = 'float'

    def format(self, v):
        return float(v)


class v4f_(VType):
    name = 'v4f'
    dims = (4,)


class v4x2f_(VType):
    name = 'v4x2f'
    dims = (4, 2)


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

    def format(self, v):
        return _format_cfg(v)


bool_ = bool__()
int32_ = int32__()
float_ = float__()
v4f = v4f_()
v4x2f = v4x2f_()
Tcfg = Tcfg_()
