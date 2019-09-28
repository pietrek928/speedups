from gnode import GNode
from proc_ctx import graph_ctx


class Loop:
    def __init__(self, start_ptr: GNode, end_ptr: GNode, shift_len: GNode, exp_use: float):
        self._start_ptr = start_ptr
        self._end_ptr = end_ptr
        self._shift_ptr = shift_len
        self._loop_ptr = start_ptr.nop()
        self._exp_use = exp_use

    def open(self):
        graph_ctx.start_use_block(self._exp_use)
        graph_ctx.append_code('do {{')
        return self._loop_ptr

    def close(self):
        graph_ctx.stationary_code('{} = {};', self._loop_ptr, self._loop_ptr + self._shift_ptr)
        graph_ctx.stationary_code('}} while ({} < {});', self._loop_ptr, self._end_ptr)
        graph_ctx.end_use_block()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
