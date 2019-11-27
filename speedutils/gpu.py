from .func import Func, FuncArg
from .gnode import LoadNode
from .proc_ctx import graph_ctx, vars_ctx
from .vtypes import int32_, bool__, Tcfg


class GpuFunc(Func):
    def __init__(self, **opts):
        super().__init__(**opts)
        self._args['ndims'] = FuncArg(
            name='ndims',
            const=True,
            type=int32_
        )

    def _process_args(self, args):
        try:
            if 'it_dims' in args or 'ndims' not in args:
                args['ndims'] = len(args['it_dims'])
        except Exception:
            raise ValueError('Could not determine `ndims`')
        return super()._process_args(args)

    def call(self, **kwargs):
        if kwargs.get('inline'):
            raise ValueError('Cannot inline gpu function')
        return super().call(**kwargs)

    @property
    def _ndims(self):
        return self.const_val('ndims', int32_)

    def _get_call_val(self, n, t):
        try:
            return vars_ctx.get(n)
        except KeyError:
            raise ValueError(f'You have to provide `{n}` for `{self._name}` call')

    @property
    def _dim_args(self):
        dim_args = tuple(
            self._get_call_val('it_dims', Tcfg)
        )
        return dim_args

    def get_pos(self, dn: int) -> LoadNode:
        raise NotImplementedError('`get_dim` not implemented')

    def get_size(self, dn: int) -> LoadNode:
        raise NotImplementedError('`get_size` not implemented')


class GroupedGpuFunc(GpuFunc):
    def _cum_pos(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_cum_pos` not implemented')

    def _cum_size(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_cum_size` not implemented')

    def _group_pos(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_group_pos` not implemented')

    def _group_size(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_group_size` not implemented')

    def _local_pos(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_local_pos` not implemented')

    def _local_size(self, dn: int) -> LoadNode:
        raise NotImplementedError('`_local_size` not implemented')

    @property
    def _nlocal_dims(self):
        return (self._ndims + 1) // 2

    def get_pos(self, dn: int) -> LoadNode:
        assert dn < self._ndims

        if dn < self._nlocal_dims:
            return self._local_pos(dn)
        else:
            return self._group_pos(dn - self._nlocal_dims)

    def get_size(self, dn: int) -> LoadNode:
        assert dn < self._ndims

        if dn < self._nlocal_dims:
            return self._group_size(dn)
        else:
            return self._local_size(dn - self._nlocal_dims)

    def cum_pos(self, dn: int):
        assert dn < self._nlocal_dims
        return self._cum_pos(dn)

    def cum_size(self, dn: int):
        assert dn < self._nlocal_dims
        return self._cum_size(dn)

    def _get_local_dims(self):
        return self._dim_args[:self._nlocal_dims]

    def _get_group_dims(self):
        group_dims = list(self._dim_args[self._nlocal_dims:])
        if len(group_dims) < self._nlocal_dims:
            group_dims.append(int32_.one())
        return group_dims


class CLExt_(bool__):
    name = 'clext'


CLExt = CLExt_()


class CLFunc(GroupedGpuFunc):
    def _cum_pos(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_id({dn})')

    def _cum_size(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_size({dn})')

    def _group_pos(self, dn: int) -> LoadNode:
        return int32_.load(f'get_group_id({dn})')

    def _group_size(self, dn: int) -> LoadNode:
        return int32_.load(f'get_num_groups({dn})')

    def _local_pos(self, dn: int) -> LoadNode:
        return int32_.load(f'get_local_id({dn})')

    def _local_size(self, dn: int) -> LoadNode:
        return int32_.load(f'get_local_size({dn})')

    def _print_exts(self, exts):
        for e in exts:
            self.codeln('#pragma OPENCL EXTENSION {} : {}'.format(
                e.name, 'enabled' if self._get_call_val(e.name, CLExt) else 'disabled'
            ))

    def gen(self, opts):
        self._print_exts(self._filter_arg_type(CLExt))
        self.codeln('__kernel')
        super().gen(opts)

    def _gen_call(self):
        name = self._get_name()
        args_fmt, args_vars = self._format_call_args(
            self._get_call_vals(
                self._var_args
            )
        )

        group_dims_fmt, group_dims_vars = self._format_call_args(
            a * b for a, b in
            zip(self._get_group_dims(), self._get_local_dims())
        )
        local_dims_fmt, local_dims_vars = self._format_call_args(
            self._get_local_dims()
        )

        graph_ctx.raw_code(
            f'{{{{'
            f'size_t group_dim_vals = {{{{ {group_dims_fmt} }}}};'
            f'size_t local_dim_vals = {{{{ {local_dims_fmt} }}}};'
            f'clSetKernelArgs(kernel, {args_fmt});'
            f'clEnqueueNDRangeKernel(queue, kernel_{name}, {self._nlocal_dims},'
            f'NULL, &group_dim_vals, &local_dim_vals, 0, NULL, NULL);'
            f'}}}}',
            *group_dims_vars, *local_dims_vars
        )


class CUDAFunc(GroupedGpuFunc):
    DIM_NAMES = 'xyz'

    def _group_pos(self, dn: int) -> LoadNode:
        return int32_.load(f'blockIdx.{self.DIM_NAMES[dn]}')

    def _group_size(self, dn: int) -> LoadNode:
        return int32_.load(f'gridDim.{self.DIM_NAMES[dn]}')

    def _local_pos(self, dn: int) -> LoadNode:
        return int32_.load(f'threadIdx.{self.DIM_NAMES[dn]}')

    def _local_size(self, dn: int) -> LoadNode:
        return int32_.load(f'blockDim.{self.DIM_NAMES[dn]}')

    def gen(self, opts):
        self.codeln('__global__')
        super().gen(opts)

    def _gen_call(self):
        name = self._get_name()
        args_fmt, arg_vars = self._format_call_args(
            self._get_call_vals(
                self._var_args
            )
        )

        group_dims_fmt, group_dims_vars = self._format_call_args(
            self._get_group_dims()
        )
        local_dims_fmt, local_dims_vars = self._format_call_args(
            self._get_local_dims()
        )

        graph_ctx.raw_code(
            f'{name}<<<dim3({group_dims_fmt}),dim3({local_dims_fmt})>>>({args_fmt});',
            *group_dims_vars, *local_dims_vars, *arg_vars
        )
