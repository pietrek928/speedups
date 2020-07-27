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


class SignOp(Op):
    def __init__(self, name, sign, args_t, **kwargs):
        args_t = tuple(
            map(lambda v: v.type_name, args_t)
        )
        args_str = 'X'.join(args_t)

        if len(args_t) == 1:
            expr = f'{sign}{{}}'
        elif len(args_t) == 2:
            expr = f'{{}}{sign}{{}}'
        else:
            raise ValueError(f'Dont know how to display sign with {len(args_t)} arguments')

        super().__init__(
            name=f'{name}Y{args_str}',
            expr=expr,
            **kwargs
        )


class TypedNoargOp(Op):
    def __init__(self, name, ret_t, **kwargs):
        super().__init__(
            name=f'{name}Y{ret_t}',
            ret_t=ret_t,
            **kwargs
        )


class ProcDescr(NamedTuple('ProcDescr', (('name', str), ('mem_levels', Tuple[MemLevel]), ('ops', Tuple[Op])))):
    pass
