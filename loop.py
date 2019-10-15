from itertools import chain
from typing import Iterable

from gnode import GNode
from proc_ctx import graph_ctx
from vtypes import MemArray, Dimension


class Loop:
    def __init__(self, exp_use: float = 1.0,
                 start_ptr: GNode = None,
                 end_ptr: GNode = None,
                 range_len: GNode = None,
                 shift_len: GNode = None,
                 external_iter: GNode = None
                 ):
        self._exp_use = exp_use

        self._shift_len = shift_len

        if external_iter is None:
            self._loop_ptr = start_ptr.nop()
        else:
            self._loop_ptr = external_iter

        if end_ptr is None:
            self._end_ptr = range_len
        else:
            self._end_ptr = end_ptr

    def open(self):
        graph_ctx.start_use_block(self._exp_use)
        graph_ctx.stationary_code('do {{')
        return graph_ctx.bind_scope(self._loop_ptr)

    def close(self):
        it = graph_ctx.bind_scope(self._loop_ptr)
        graph_ctx.stationary_code('{} = {};', it, it + self._shift_len)
        graph_ctx.stationary_code('}} while ({} < {});', it, self._end_ptr)
        graph_ctx.end_use_block()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ArraysLoop:
    def __init__(self, arrs: Iterable[MemArray], dim_order: Iterable[Dimension]):
        self._arrs = tuple(arrs)
        self._dim_order = tuple(dim_order)

        adims = tuple(
            a.ddims for a in self._arrs
        )
        dim_ns = set(chain(*(
            a.ddims.keys() for a in self._arrs
        )))
        for dim in dim_ns:
            vs = set(
                dd[dim] for dd in adims
                if dd[dim] > 1
            )
            if len(vs) > 1:
                raise ValueError(f'Insufficient values {vs} for dimension {dim}')
