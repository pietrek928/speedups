from .func import Func, FuncArg
from .gnode import LoadNode
from .proc_ctx import graph_ctx
from .vtypes import int32_, bool__


class GpuFunc(Func):
    def call(self, **kwargs):
        if kwargs.get('inline'):
            raise ValueError('Cannot inline gpu function')
        return super().call(**kwargs)

    @property
    def _dim_args(self):
        return tuple(
            FuncArg(name=f'size{dn}', const=False, type=int32_)
            for dn in range(self.const_val('ndims', int32_))
        )


class CLExt_(bool__):
    name = 'clext'


CLExt = CLExt_()


class CLFunc(GpuFunc):
    def get_dim(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_id({dn})')

    def get_size(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_size({dn})')

    def _print_exts(self, exts):
        for e in exts:
            self.codeln('#pragma OPENCL EXTENSION {} : {}'.format(
                e.name, 'enabled' if self.const_val(e.name, CLExt) else 'disabled'
            ))

    def gen(self, opts):
        self._print_exts(self._filter_arg_type(CLExt))
        self.codeln('__kernel')
        super().gen(opts)


class CUDAFunc(GpuFunc):
    DIM_NAMES = 'xyz'

    def _block_id(self, dn: int) -> LoadNode:
        return int32_.load(f'blockIdx.{self.DIM_NAMES[dn]}')

    def _block_dim(self, dn: int) -> LoadNode:
        return int32_.load(f'blockDim.{self.DIM_NAMES[dn]}')

    def _thread_id(self, dn: int) -> LoadNode:
        return int32_.load(f'threadIdx.{self.DIM_NAMES[dn]}')

    def _grid_dim(self, dn: int) -> LoadNode:
        return int32_.load(f'gridDim.{self.DIM_NAMES[dn]}')

    # def get_dim_pos(self, dn: int) -> LoadNode:
    #     return self._block_id(dn) * self._block_dim(dn) + self._thread_id(dn)

    def get_dim(self, dn: int) -> LoadNode:
        ndims = self.const_val('ndims', int32_)
        if dn < ndims:
            return self._thread_id(dn)
        else:
            return self._block_id(dn - ndims)  # TODO: mul ?????

    def get_size(self, dn: int) -> LoadNode:
        ndims = self.const_val('ndims', int32_)
        if dn < ndims:
            return self._block_dim(dn)
        else:
            return self._grid_dim(dn - ndims)  # TODO: mul ?????

    def gen(self, opts):
        self.codeln('__global__')
        super().gen(opts)

    def _gen_call(self):
        name = self._get_name()
        dim_args = self._dim_args
        args_fmt, arg_vars = self._format_call_args(self._var_args)

        block_dims_fmt, block_dims_vars = self._format_call_args(dim_args[:len(dim_args) // 2])
        grid_dims_fmt, grid_dims_vars = self._format_call_args(dim_args[len(dim_args) // 2:])

        graph_ctx.raw_code(
            f'{name}<<<dim3({grid_dims_fmt}),dim3({block_dims_fmt})>>>({args_fmt})',
            *grid_dims_vars, *block_dims_vars, *arg_vars
        )
