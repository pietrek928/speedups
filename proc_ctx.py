from _contextvars import ContextVar
from contextlib import contextmanager

from flow import FlowGraph
from proc_descr import ProcDescr


class ProcCtx:
    def __init__(self, pd: ProcDescr):
        self._pd = pd

    def new_graph(self):
        return FlowGraph(mem_levels=self._pd.mem_levels, ops=self._pd.ops)

    @property
    def arch(self):
        return self._pd.name


_proc_ctx = ContextVar('proc_ctx')


class Ctx:
    def __getattr__(self, n):
        return getattr(_proc_ctx.get(), n)


@contextmanager
def proc(pd: ProcDescr):
    old = _proc_ctx.set(ProcCtx(pd))
    try:
        yield
    finally:
        _proc_ctx.reset(old)


proc_ctx = Ctx()
