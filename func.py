from typing import NamedTuple, Dict, List, Any, Tuple, Iterable

from .gnode import GNode, LoadNode
from .proc_ctx import proc_ctx, new_graph, graph_ctx, vars_ctx, use_vars, func_scope, use_only_vars
from .vtypes import VType, bool_, int32_


class FuncArg(
    NamedTuple('func_param', (('name', str), ('const', bool), ('type', VType)))
):
    pass


func_reg = {}


class Func:
    _consts = ()

    def __init__(self, name=None, **opts):
        self._name = name
        self._opts = opts
        self._args: Dict[str, FuncArg] = {}

    def _print(self, l):
        print(l)

    def _filter_arg_type(self, t: VType):
        return sorted(
            a for a in self._args.values() if a.type == t
        )

    @property
    def _var_args(self) -> List[FuncArg]:
        return sorted(
            a for a in self._args.values() if not a.const
        )

    @property
    def _const_args(self) -> List[FuncArg]:
        return sorted(
            a for a in self._args.values() if a.const
        )

    def _get_name(self) -> str:
        const_descr = 'X'.join(
            f'{k}V{t.format(v)}' for k, t, v in self._get_consts()
        ).replace('.', 'K')
        return f'{self._name}_' \
               f'{const_descr}_' \
               f'F{proc_ctx.arch}'

    def _gen_body(self):
        with new_graph() as graph:
            self._func()  # TODO: args by signature ?
            graph.gen_code()

    def _analyze(self):
        with new_graph():
            self._func()

    def gen(self, opts):
        with func_scope(self):
            with use_only_vars(**self._get_all_opts(opts)):
                self._analyze()

            with use_only_vars(**self._get_all_opts(opts)):
                self._print('void {}({}) {{'.format(
                    self._get_name(),
                    ', '.join(f'{arg.type} {arg.name}' for arg in self._var_args)
                ))
                self._gen_body()
                self._print('}')

    def _lookup_var(self, name, t: VType, const=False, default=None) -> Tuple[FuncArg, Any]:
        if name not in self._args:
            a = FuncArg(
                name=name,
                const=const,
                type=t
            )
        else:
            a = self._args[name]
            if a.type != t:
                raise ValueError(f'Invalid type `{t}` for `{name}` having already type `{a.type}`')

        if default is not None:
            self._opts[name] = t.format(default)  # TODO: copy obj ?

        if const:
            if not a.const:
                a = FuncArg(
                    name=a.name,
                    const=True,
                    type=t
                )
            v = vars_ctx.get(name)  # !!!!!!!!!
            if v is not None:
                val = v
            else:
                val = self._opts.get(name)
        else:
            val = a.name

        if val is None:
            raise ValueError(f'Value for `{name}` was not provided to `{self._name}`')

        self._args[name] = a

        return a, val

    def v(self, v, t: VType):
        if isinstance(v, str):
            return self.get_var(v, t)
        else:
            return graph_ctx.const(t, v)

    def get_var(self, name, t: VType, const=False, default=None) -> GNode:
        a, val = self._lookup_var(name, t, const=const, default=default)

        if a.const:
            return graph_ctx.const(t, val)
        else:
            return graph_ctx.var(t, name, start_scope=True)

    def const_val(self, name, t: VType, default=None):
        a, val = self._lookup_var(name, t, const=True, default=default)

        if not a.const:
            raise ValueError(f'Variable {name} is not const')

        return val

    def _get_all_opts(self, args):
        r = dict(self._opts)
        r.update(args)
        return r

    def _format_call_args(self, arg_list: Iterable[FuncArg]) -> Tuple[str, List[GNode]]:
        call_args = []
        arg_nodes = []
        for arg in arg_list:
            v = vars_ctx.get(arg.name)
            if v is None:
                raise ValueError(f'You must provide {arg.name} to {self._name}')
            if isinstance(v, GNode):
                call_args.append('{}')
                arg_nodes.append(v)
            else:
                call_args.append(arg.type.format(v))
        return ', '.join(str(v) for v in call_args), arg_nodes

    def _get_consts(self) -> List[Tuple[str, VType, Any]]:
        consts = []
        for const in self._const_args:
            v = vars_ctx.get(const.name)
            if v is None:
                raise ValueError(f'You must provide {const.name} to {self._name}')
            consts.append((const.name, const.type, v))
        return sorted(consts)

    def _register(self):
        global func_reg

        name = self._get_name()
        if name not in func_reg:
            func_reg[name] = (self, self._get_consts())

    def _check_registered(self):
        global func_reg
        return self._get_name() in func_reg

    def _gen_call(self):
        name = self._get_name()
        args_fmt, arg_vars = self._format_call_args(self._var_args)
        graph_ctx.raw_code(f'{name}({args_fmt})', *arg_vars)

    def call(self, **kwargs):
        inline = kwargs.pop('inline', False)
        opts = self._get_all_opts(kwargs)  # TODO: auto pick vars
        if not inline:
            with func_scope(self):
                with use_only_vars(**opts):
                    registered = self._check_registered()
                    if not registered:
                        self._analyze()
                if not registered:
                    opts = self._get_all_opts(kwargs)
                with use_only_vars(**opts):
                    if not registered:
                        self._register()
                    self._gen_call()
        else:
            with use_vars(**opts):
                self._func()

    def decor(self, f):  # TODO: copy object ?
        self._func = f
        if not self._name:
            self._name = self._func.__name__
        return self

    def __call__(self, *args, **kwargs):
        if not kwargs and len(args) == 1 and callable(args[0]):
            return self.decor(args[0])
        else:
            assert not args
            return self.call(**kwargs)


# class LoopFunc(Func):
#     def gen_body(self, args):
#         for v in self.get_arg('loop_dims')

# def _merge_args_vals(args_descr: Dict[str, FuncArg], args):
#     return {
#         n: args.get(n, args_descr[n].default)
#         for n in args_descr.keys()
#     }


class GpuFunc(Func):
    def call(self, **kwargs):
        if kwargs.get('inline'):
            raise ValueError('Cannot inline gpu function')
        return super().call(**kwargs)

    @property
    def _dim_args(self):
        return tuple(
            FuncArg(name=f'size{dn}', const=False, type=int32_)
            for dn in range(self._opts['ndims'])
        )


class CLExt_(VType):
    name = 'clext'

    def format(self, v):
        return bool(v)


CLExt = CLExt_()


class CLFunc(GpuFunc):
    def get_dim(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_id({dn})')

    def get_size(self, dn: int) -> LoadNode:
        return int32_.load(f'get_global_size({dn})')

    def _print_exts(self, exts):
        for e in exts:
            self._print('#pragma OPENCL EXTENSION {} : {}'.format(
                e.name, 'enabled' if self.get_var(e.name, bool_, const=True) else 'disabled'
            ))

    def gen(self, opts):
        self._print_exts(self._filter_arg_type(CLExt))
        self._print('__kernel')
        super().gen(opts)


class CUDAFunc(GpuFunc):
    DIM_NAMES = 'xyz'

    def _block_id(self, dn: int) -> LoadNode:
        return int32_.load(f'blockIdx.{self.DIM_NAMES[dn]}')

    def _block_dim(self, dn: int) -> LoadNode:
        return int32_.load(f'blockDim.{self.DIM_NAMES[dn]}')

    def _thread_id(self, dn: int) -> LoadNode:
        return int32_.load(f'threadIdx.{self.DIM_NAMES[dn]}')

    def get_dim_pos(self, dn: int) -> LoadNode:
        return self._block_id(dn) * self._block_dim(dn) + self._thread_id(dn)

    def get_size(self, dn: int) -> LoadNode:  # ???????
        dim_name = self.DIM_NAMES[dn]
        return int32_.load(
            'blockDim.{} * blockIdx.{}'.format(
                dim_name, dim_name
            )
        )

    def gen(self, opts):
        self._print('__global__')
        super().gen(opts)

    def _gen_call(self):
        name = self._get_name()
        dims_fmt, dims_vars = self._format_call_args(self._dim_args)
        args_fmt, arg_vars = self._format_call_args(self._var_args)
        graph_ctx.raw_code(f'{name}<<<{dims_fmt}>>>({args_fmt})', *dims_vars, *arg_vars)
