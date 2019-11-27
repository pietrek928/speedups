from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Iterable, Type

from .array import MemArray, Dimension, CumDims
from .func import Func
from .gnode import GNode
from .gpu import GroupedGpuFunc, GpuFunc
from .proc_ctx import graph_ctx, func_ctx
from .vtypes import int32_, Tcfg


def _it_name(dname) -> str:
    return f'it_{dname}'


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
        @wraps(f)
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
                it_n = _it_name(n)
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


def _create_gpu_loop_func(gpu_func_t: Type[GpuFunc]) -> Type[GpuFunc]:
    class loop_t(gpu_func_t):
        def __init__(self, **opts):
            super().__init__(**opts)

        def _process_args(self, args):
            if 'loop_dims' in args:
                loop_dims = list(reversed(args.pop('loop_dims')))

                loop_cfg = []

                nn = len(loop_dims)
                nblock_dims = (nn + 1) // 2  # TODO: if
                it_dims = [1] * (nblock_dims * 2)

                it = 0
                while it < len(loop_dims):
                    dd = loop_dims[it]
                    if dd is not None:
                        n, sz = dd
                        s = [sz]
                        it_dims[it] = sz  # !!!! div ?
                        if isinstance(func_ctx, GroupedGpuFunc) and it < nblock_dims:
                            for i in range(it + 1, len(loop_dims)):
                                if loop_dims[i] is not None:
                                    dn, ds = loop_dims[i]
                                    if dn == n:
                                        s.append(ds)  # !!!! div ?
                                        loop_dims[i] = loop_dims[it + nblock_dims]
                                        loop_dims[it + nblock_dims] = None
                                        it_dims[it + nblock_dims] = ds  # !!!! div ?
                                        break
                        loop_cfg.append((n, it, tuple(s)))
                    it += 1

                args['it_dims'] = tuple(it_dims)
                args['loop_cfg'] = tuple(loop_cfg)
            return super()._process_args(args)

        def decor(self, f: Func):
            @wraps(f)
            def wrapper(f: Func = f):
                loop_cfg = func_ctx.const_val('loop_cfg', Tcfg)
                block_ddims = func_ctx.const_val('block_ddims', Tcfg)

                iters = defaultdict(int32_.zero)
                sizes = defaultdict(int32_.one, {
                    _it_name(n): int32_(s)
                    for n, s in block_ddims
                })

                for n, it, sz in loop_cfg:
                    itn = _it_name(n)
                    pos_shift = sizes[itn]
                    sizes[itn] *= int32_(sz[0])
                    if len(sz) == 1:
                        pos = func_ctx.get_pos(it)
                    else:
                        pos = func_ctx.cum_pos(it)
                        sizes[itn] *= int32_(sz[1])
                    iters[itn] += pos_shift * pos

                return f(**iters)

            return super().decor(wrapper)

    loop_t.__name__ += f'_{loop_t.__name__}'
    return loop_t


GPU_LOOP_FUNCS = {}


def GpuLoopFunc(gpu_func_t: Type[GpuFunc], **kwargs):
    global GPU_LOOP_FUNCS
    try:
        loop_t = GPU_LOOP_FUNCS[gpu_func_t]
    except KeyError:
        loop_t = _create_gpu_loop_func(gpu_func_t)
    return loop_t(**kwargs)


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
