from _contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .flow import FlowGraph
    from .func import Func


class ProcCtx:
    def __init__(self, pd):
        self._pd = pd

    def new_graph(self) -> 'FlowGraph':
        from .flow import FlowGraph
        return FlowGraph(mem_levels=self._pd.mem_levels, ops=self._pd.ops)

    @property
    def arch(self):
        return self._pd.name


class CtxVarProxy:
    def __init__(self, n):
        self._var = ContextVar(n)
        self._get = self._var.get
        self._set = self._var.set
        self.reset = self._var.reset

    @property
    def value(self):
        return self._get()

    def __getattr__(self, n):
        return getattr(self._get(), n)


class VarsCtx(CtxVarProxy):
    def __init__(self, n):
        super().__init__(n)

    def set_vars(self, **kwargs):
        new_dict = dict(self._get(()))
        new_dict.update(**kwargs)
        return self._set(new_dict)

    def reset_vars(self, **kwargs):
        return self._set(kwargs)

    def get(self, n, default=None):
        return self._get().get(n, default)

    def __getitem__(self, n):
        return self.get(n)

    def __getattr__(self, n):
        return self.get(n)


proc_ctx: Union[ProcCtx, CtxVarProxy] = CtxVarProxy('proc_ctx')
graph_ctx: Union['FlowGraph', CtxVarProxy] = CtxVarProxy('graph_ctx')
func_ctx: Union['Func', CtxVarProxy] = CtxVarProxy('func_ctx')
vars_ctx = VarsCtx('vars_ctx')


@contextmanager
def proc(pd):
    old = proc_ctx._set(ProcCtx(pd))
    try:
        yield pd
    finally:
        proc_ctx.reset(old)


@contextmanager
def func_scope(f):
    old = func_ctx._set(f)
    try:
        yield f
    finally:
        func_ctx.reset(old)


@contextmanager
def new_graph() -> Iterable['FlowGraph']:
    graph = proc_ctx.new_graph()
    old = graph_ctx._set(graph)
    try:
        yield graph
    finally:
        graph_ctx.reset(old)


@contextmanager
def set_vars(**kwargs):
    old = vars_ctx.set_vars(**kwargs)
    try:
        yield
    finally:
        vars_ctx.reset(old)


@contextmanager
def clear_vars(**kwargs):
    old = vars_ctx.reset_vars(**kwargs)
    try:
        yield
    finally:
        vars_ctx.reset(old)
