from itertools import chain
from typing import NamedTuple, Dict

from proc_ctx import proc_ctx
from vtypes import VType, bool_, int32_


class FuncArg(
    NamedTuple('func_param', (('name', str), ('const', bool), ('type', VType)))
):
    pass


def _is_var(v):
    return isinstance(v, str)


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
    def var_args(self):
        return sorted(
            a for a in self._args.values() if not a.const
        )

    @property
    def const_args(self):
        return sorted(
            a for a in self._args.values() if a.const
        )

    def get_name(self, opts):
        return '{}_{}'.format(
            self._name,
            '_'.join(chain(
                ('{}x{}'.format(k, v) for k, v in sorted(
                    self.get_consts(opts).items()
                )),
                ('F{}'.format(proc_ctx.arch), )
            ))
        )

    def gen_body(self, opts):
        self._graph = proc_ctx.new_graph()
        self._func(**opts, ctx=self)
        self._graph.gen_code()

    def analyze(self, opts):
        self._graph = proc_ctx.new_graph()
        self._func(**opts, ctx=self)

    def gen(self, opts):
        self.analyze(opts)
        self._print('void {}({}) {{'.format(
            self.get_name(opts),
            ', '.join('{} {}'.format(arg.type, arg.name) for arg in self.var_args)
        ))
        self.gen_body(opts)
        self._print('}')

    def get_val(self, name, t, const=False, default=None):
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

        if default is not None and name not in self._opts:
            self._opts[name] = default  # TODO: copy obj ?

        if const:
            if not a.const:
                a = FuncArg(
                    name=a.name,
                    const=True,
                    type=t
                )
            try:
                r = self._graph.const(t, self._opts[name])
            except KeyError:
                raise ValueError('Value for {} was not provided to {}'.format(name, self._name))
        else:
            r = self._graph.var(t, name)

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

    def format_call_args(self, args):
        call_args = []
        for arg in self.var_args:
            if arg.name in args:
                v = args[arg.name]
            else:
                v = self._opts.get(arg.name)
            if v is None:
                raise ValueError('You must provide {} to {}', arg.name, self._name)
            call_args.append(str(v))
        return ', '.join(call_args)

    def get_consts(self, args):
        consts = {}
        for const in self.const_args:
            if const.name in args:
                v = args[const.name]
            else:
                v = self._opts.get(const.name)
            if v is None:
                raise ValueError('You must provide {} to {}', const.name, self._name)
            consts[const.name] = const.type.format(v)
        return consts

    def register(self, opts):
        global func_reg

        name = self.get_name(opts)
        if name not in func_reg:
            func_reg[name] = (self, self.get_consts(opts))

    def call(self, **args):
        self.register(args)
        name = self.get_name(args)
        self._print('{}({});'.format(name, self.format_call_args(args)))

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
    def get_dim(self, dn):
        return self.get_val('__dim_' + dn, int32_)

    def _print_exts(self, exts):
        for e in exts:
            self._print('#pragma OPENCL EXTENSION {} : {}'.format(
                e.name, 'enabled' if self.get_val(e.name, bool_) else 'disabled'
            ))

    def gen(self, opts):
        self._print_exts(self._filter_arg_type(CLExt))
        self._print('__kernel')
        super().gen(opts)


class CUDAFunc(Func):
    def get_dim(self, dn):
        return self.get_val('__dim_' + dn, int32_)

    def gen(self, opts):
        self._print('__global__')
        super().gen(opts)

    # def call(self, x=, y=, z=, **opts):


