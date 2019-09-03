from typing import NamedTuple, Tuple, Optional

from vtypes import VType

MemLevel = NamedTuple('MemLevel', (('name', str), ('size', int), ('port_n', int), ('load_time', float)))
Op = NamedTuple('Op', (
    ('name', str), ('ret_t', Optional[VType]), ('exec_t', float), ('ports', Tuple[int]), ('args_ardered', bool)
))


class ProcDescr(NamedTuple('ProcDescr', (('name', str), ('mem_levels', Tuple[MemLevel]), ('ops', Tuple[Op])))):
    pass
