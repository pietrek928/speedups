from itertools import chain
from typing import NamedTuple, Dict, List, Any

from gnode import GNode, LoadNode
from proc_ctx import proc_ctx, new_graph, graph_ctx, vars_ctx, use_vars
from vtypes import VType, bool_, int32_


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
        return '{}_{}'.format(
            self._name,
            '_'.join(chain(
                ('{}x{}'.format(k, v) for k, v in sorted(
                    self._get_consts().items()
                )),
                ('F{}'.format(proc_ctx.arch),)
            ))
        )

    def _gen_body(self):
        with new_graph() as graph:
            self._func(ctx=self)  # TODO: args by signature ?
            graph.gen_code()

    def _analyze(self):
        with new_graph():
            self._func(ctx=self)

    def gen(self, opts):
        with use_vars(**self._get_all_opts(opts)):
            self._analyze()
            self._print('void {}({}) {{'.format(
                self._get_name(),
                ', '.join('{} {}'.format(arg.type, arg.name) for arg in self._var_args)
            ))
            self._gen_body()
            self._print('}')

    def get_val(self, name, t, const=False, default=None) -> GNode:
        if name not in self._args:
            a = FuncArg(
                name=name,
                const=const,
                type=t
            )
        else:
            a = self._args[name]
            if a.type != t:
                raise ValueError('Invalid type {} for {} having already type {}'.format(t, name, a.type))

        if default is not None:
            self._opts[name] = default  # TODO: copy obj ?

        if const:
            if not a.const:
                a = FuncArg(
                    name=a.name,
                    const=True,
                    type=t
                )
            v = vars_ctx.get(name)
            if v is not None:
                r = graph_ctx.const(t, v)
            else:
                raise ValueError('Value for {} was not provided to {}'.format(name, self._name))
        else:
            r = graph_ctx.var(t, name, start_scope=True)

        self._args[name] = a

        return r

    # def loop(self, start, step, size):
    #     vname = 'it' + str(self._it_num)
    #     self._it_num += 1
    #
    #     self._print('do {')
    #     self._print('auto {} = ({});'.format(vname, start))
    #     self._print('auto {}_end = {} + ({});'.format(vname, vname, size))
    #
    #     self._stack.append([
    #         '{} += {};'.format(vname, step),
    #         '}} while ({} < {}_end);'.format(vname, vname)
    #     ])
    #
    #     return vname
    #
    # def loop_end(self):
    #     for l in self._stack.pop():
    #         self._print(l)

    def _get_all_opts(self, args):
        r = dict(self._opts)
        r.update(args)
        return r

    def _format_call_args(self) -> str:
        call_args = []
        for arg in self._var_args:
            v = vars_ctx.get(arg.name)
            if v is None:
                raise ValueError('You must provide {} to {}', arg.name, self._name)
            call_args.append(arg.type.format(v))
        return ', '.join(str(v) for v in call_args)

    def _get_consts(self) -> Dict[str, Any]:
        consts = {}
        for const in self._const_args:
            v = vars_ctx.get(const.name)
            if v is None:
                raise ValueError('You must provide {} to {}', const.name, self._name)
            consts[const.name] = const.type.format(v)
        return consts

    def _register(self):
        global func_reg

        name = self._get_name()
        if name not in func_reg:
            func_reg[name] = (self, self._get_consts())

    def call(self, **args):
        with use_vars(**self._get_all_opts(args)):
            self._analyze()
            self._register()
            name = self._get_name()
            self._print('{}({});'.format(name, self._format_call_args()))

    def __call__(self, *args, **kwargs):
        if not kwargs and len(args) == 1 and callable(args[0]):
            self._func = args[0]  # TODO: copy ?
            if self._name is None:
                self._name = self._func.__name__
            return self
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


class CLExt_(VType):
    name = 'clext'

    def format(self, v):
        return bool(v)


CLExt = CLExt_()


class CLFunc(Func):
    def get_dim(self, dn: int) -> LoadNode:
        return int32_.load('get_global_id({})'.format(dn))

    def get_size(self, dn: int) -> LoadNode:
        return int32_.load('get_global_size({})'.format(dn))

    def _print_exts(self, exts):
        for e in exts:
            self._print('#pragma OPENCL EXTENSION {} : {}'.format(
                e.name, 'enabled' if self.get_val(e.name, bool_, const=True) else 'disabled'
            ))

    def gen(self, opts):
        self._print_exts(self._filter_arg_type(CLExt))
        self._print('__kernel')
        super().gen(opts)


class CUDAFunc(Func):
    DIM_NAMES = 'xyz'

    def get_dim(self, dn) -> LoadNode:
        dim_name = self.DIM_NAMES[dn]
        return int32_.load(
            'blockIdx.{} * blockDim.{} + threadIdx.{}'.format(
                dim_name, dim_name, dim_name
            )
        )

    def get_size(self, dn: int) -> LoadNode:
        dim_name = self.DIM_NAMES[dn]
        return int32_.load(
            'blockDim.{} * blockIdx.{}'.format(
                dim_name, dim_name
            )
        )

    def gen(self, opts):
        self._print('__global__')
        super().gen(opts)

    # def call(self, x=, y=, z=, **opts):
