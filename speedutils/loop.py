from itertools import chain
from typing import Iterable

from .array import MemArray, Dimension, CumDims
from .func import Func
from .gnode import GNode
from .gpu import GpuFunc
from .proc_ctx import graph_ctx, func_ctx
from .vtypes import int32_, Tcfg


class Loop:
    def __init__(
            self,
            exp_use: float = 1.0,
            start_val: GNode = None,
            end_val: GNode = None,
            range_len: GNode = None,
            shift_len: GNode = None,
            external_iter: GNode = None
    ):
        self._exp_use = exp_use

        self._shift_len = shift_len

        if external_iter is None:
            self._iter_val = graph_ctx.sep(start_val)
        else:
            self._iter_val = external_iter

        if end_val is None:
            self._end_val = range_len
        else:
            self._end_val = end_val

    def open(self):
        graph_ctx.start_use_block(self._exp_use)
        graph_ctx.stationary_code('do {{')
        return graph_ctx.bind_scope(self._iter_val)

    def close(self):
        it = graph_ctx.bind_scope(self._iter_val)
        graph_ctx.stationary_code('{} = {};', it, it + self._shift_len)
        graph_ctx.stationary_code('}} while ({} < {});', it, self._end_val)
        graph_ctx.end_use_block()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LoopFunc(Func):
    def decor(self, f: Func):
        def wrapper(f: Func = f):
            loop_dims = func_ctx.const_val('loop_dims', Tcfg)
            block_ddims = CumDims.from_cfg(
                func_ctx.const_val('block_ddims', Tcfg)
            )
            block_sizes = []
            for d in loop_dims:
                block_sizes.append(block_ddims)
                block_ddims *= CumDims.from_cfg([d])

            loop_stack = []
            iters = {}
            for d, s in zip(reversed(loop_dims), reversed(block_sizes)):
                d = Dimension(*d)
                n = d.n
                it_n = f'it_{n}'
                l = Loop(
                    start_val=iters[it_n] if it_n in iters else int32_.zero(),
                    range_len=int32_.v(d.size),
                    shift_len=int32_.v(s[n])
                )
                iters[it_n] = l.open()
                loop_stack.append(l)
            r = f(**iters)  # TODO: func in name
            while loop_stack:
                loop_stack.pop().close()
            return r

        return super().decor(wrapper)


class GpuLoopFunc(Func):
    def decor(self, f: GpuFunc):
        def wrapper(f: Func = f):
            loop_dims = func_ctx.const_val('loop_dims', Tcfg)
            block_ddims = CumDims.from_cfg(
                func_ctx.const_val('block_ddims', Tcfg)
            )

        return super().decor(wrapper)


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

    # def get_ranges(self):
