from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

from .vtypes import VType

MemLevel = NamedTuple('MemLevel', (('name', str), ('size', int), ('port_n', int), ('load_time', float)))


@dataclass
class Op:
    name: str
    ret_t: Optional[VType]
    exec_t: float
    ports: Tuple[int]
    args_ordered: bool = True
    expr: str = ''

    def format_expr(self, args):
        if self.expr:
            return self.expr.format(*args)
        args_str = ', '.join(args)
        return f'{self.name}({args_str})'


class FuncOp(Op):
    def __init__(self, name, args_t, **kwargs):
        args_str = 'X'.join(
            a.type_name for a in args_t
        )
        super().__init__(
            name=f'{name}Y{args_str}',
            **kwargs
        )


class SimpleLoadOp(Op):
    def __init__(self, ret_t: VType, **kwargs):
        super().__init__(
            name=f'loadY{ret_t.type_name}',
            ret_t=ret_t,
            expr='({})',
            **kwargs
        )


class SimpleStoreOp(Op):
    def __init__(self, in_t: VType, **kwargs):
        super().__init__(
            name=f'storY{in_t.type_name}',
            ret_t=None,
            expr='({})',
            **kwargs
        )


class SignOp(Op):
    def __init__(self, name, sign, args_t, args_ordered=True, **kwargs):
        args_t = tuple(
            v.type_name for v in args_t
        )
        if not args_ordered:
            args_t = sorted(args_t)
        args_str = 'X'.join(args_t)

        if len(args_t) == 1:
            expr = f'{sign} {{}}'
        elif len(args_t) == 2:
            expr = f'{{}} {sign} {{}}'
        else:
            raise ValueError(f'Dont know how to display sign with {len(args_t)} arguments')

        super().__init__(
            name=f'{name}Y{args_str}',
            expr=expr,
            args_ordered=args_ordered,
            **kwargs
        )


class CvtOp(Op):
    def __init__(self, in_t: VType, ret_t: VType, expr=None, **kwargs):
        if expr is None:
            expr = f'{ret_t.type_name}({{}})'
        super().__init__(
            name=f'cvtY{in_t.type_name}X{ret_t.type_name}',
            ret_t=ret_t,
            expr=expr,
            **kwargs
        )


class TypedNoargOp(Op):
    def __init__(self, name, ret_t, **kwargs):
        super().__init__(
            name=f'{name}Y{ret_t.type_name}',
            ret_t=ret_t,
            **kwargs
        )


class ProcDescr(NamedTuple('ProcDescr', (('name', str), ('mem_levels', Tuple[MemLevel]), ('ops', Tuple[Op])))):
    pass
