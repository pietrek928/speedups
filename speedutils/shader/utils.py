from _contextvars import ContextVar
from inspect import getfullargspec
from typing import Iterable, List, Optional, Tuple

from speedutils.graphval import GraphVal, VType
from .types import Float, convert_arg
from ..proc_ctx import graph_ctx, new_graph


class ShaderInputProperty:
    _name = None
    _args = ()

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            self._set_func(args[0])
        else:
            self._setup(*args, **kwargs)

    def __call__(self, func):
        self._set_func(func)
        return self

    def _set_func(self, func):
        self._func = func
        if not self._name:
            self._name = func.__name__
        argspec = getfullargspec(func)
        func_args = tuple(argspec.args)
        assert func_args[0] == 'self'
        self._args = func_args[1:]

    def _setup(self, name=None, func=None):
        if func is not None:
            self._name = name
            self._set_func(func)

    def __get__(self, shader: 'Shader', objtype=None):
        if shader is None:
            return self

        func_kwargs = {
            n: getattr(shader, n)
            for n in self._args
        }
        return self._func(shader, **func_kwargs)

    def __set__(self, shader: 'Shader', value):
        raise ValueError(f'Cannot set shader input property `{self._name}` for `{type(shader).__name__}`')


class ShaderOutputProperty:
    _name = None
    _type = None
    _func = None
    _register = True

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            self._set_func(args[0])
        else:
            self._setup(*args, **kwargs)

    def __call__(self, func):
        self._set_func(func)
        return self

    def _set_func(self, func):
        self._func = func
        if not self._name:
            self._name = func.__name__
        argspec = getfullargspec(func)
        func_args = tuple(argspec.args)
        assert func_args[0] == 'self'

    def _setup(self, name=None, func=None, type: Optional[VType] = None, register=True):
        self._name = name
        if func is not None:
            self._set_func(func)
        self._type = type
        self._register = register

    def __get__(self, shader: 'Shader', objtype=None):
        raise ValueError(f'Cannot get shader output property `{self._name}` from `{type(shader).__name__}`')

    def __set__(self, shader: 'Shader', value):
        if shader is None:
            raise ValueError(f'Setting property {self._name} on class is not allowed')

        value = convert_arg(value)

        if self._func is not None:
            value = self._func(shader, value)

        if self._type is not None:
            value = value.cvt(self._type)
            # if not value.istype(self._type):
            #     raise ValueError(f'Output property {self._name} should have type `{self._type}`, but got {value.type}')

        shader.out_(self._name, value, register=self._register)


_shader_context: ContextVar['Shader'] = ContextVar('gen_context', default=None)


def get_context_shader() -> 'Shader':
    shader = _shader_context.get(None)
    assert shader is not None, 'No shader in current context'
    return shader


class Shader:
    def __init__(self, inputs: Iterable[GraphVal] = (), uniform_inputs: Iterable[GraphVal] = (), version=330):
        self._version = version
        self._input: List[GraphVal] = list(inputs)
        self._uniform_input: List[GraphVal] = list(uniform_inputs)
        self._output: List[GraphVal] = []
        self._code_lines = []

    def supports_version(self, v: int):
        return v <= self._version

    @property
    def uniform_params(self) -> Tuple[GraphVal]:
        return tuple(self._uniform_input)

    @property
    def output_params(self) -> Tuple[GraphVal]:
        return tuple(self._output)

    @staticmethod
    def find_var(vars: Iterable[GraphVal], n, t=None) -> Optional[GraphVal]:
        for var in vars:
            if var.name == n:
                if t is not None:
                    if not var.istype(t):
                        raise ValueError(f'Input variable `{n}` with type {t} requested, but has type {var.type}')
                return var

    def in_(self, n, t: VType = None) -> GraphVal:
        raise NotImplementedError(f'Input unimplemented for shader {self.__class__}')

    def in_uni(self, n: str, t: VType = Float):
        v = self.find_var(self._uniform_input, n, t=t)
        if v is None:
            v = t.var(n, start_scope=True)
            self._uniform_input.append(v)
        return v

    def out_(self, n: str, v: GraphVal, register=True):
        if self.find_var(self._output, n) is not None:
            raise ValueError(f'Output var `{n}` already set')
        v.from_expr(
            f'{n} = {{}}', v,
            op=graph_ctx.find_store_op(type(v))
        )
        if register:
            self._output.append(v)

    # def gen_var_name(self, prefix=None):
    #     prefix = prefix or 'v'
    #     n = f'{prefix}{self._name_counter[prefix]}'
    #     self._name_counter[prefix] += 1
    #     return n

    # def add_function_call(self, rt: VType, func_name, *func_args):
    #     rv = rt()
    #     call_args = ', '.join(map(str, func_args))
    #     self._code_lines.append(
    #         f'{rv.typename} {rv} = {func_name}({call_args});'
    #     )
    #     return rv
    #
    # def add_operator_call(self, rt: Type[Var], op_name, op_arg1, op_arg2):
    #     rv = rt()
    #     self._code_lines.append(
    #         f'{rv.typename} {rv} = {op_arg1} {op_name} {op_arg2};'
    #     )
    #     return rv

    def _gen_code(self):
        pass

    def _render_uniforms(self, fixed_location=True):
        for i, uni_v in enumerate(self._uniform_input):
            if fixed_location and self.supports_version(430):
                yield f'layout(location = {i}) uniform {uni_v.typename} {uni_v};'
            else:
                yield f'uniform {uni_v.typename} {uni_v};'

    def _render_in(self, fixed_location=True):
        for i, in_v in enumerate(self._input):
            if fixed_location and self.supports_version(430):
                yield f'layout(location = {i}) in {in_v.typename} {in_v};'
            else:
                yield f'in {in_v.typename} {in_v};'

    def _render_out(self):
        for out_v in self._output:
            yield f'out {out_v.typename} {out_v};'

    def _render_definitions(self):
        return ()

    def _render_content(self):
        with new_graph() as g:
            self._gen_code()
            # g.print_graph()
            g.gen_code()
        yield f'#version {self._version}'
        yield from self._render_definitions()
        yield 'void main() {'
        yield from self._code_lines
        yield '}'

    def gen(self):
        old_val = _shader_context.set(self)
        try:
            return '\n'.join(self._render_content())
        finally:
            _shader_context.reset(old_val)
