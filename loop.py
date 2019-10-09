from gnode import GNode
from proc_ctx import graph_ctx


class Loop:
    def __init__(self, exp_use: float = 1.0,
                 start_ptr: GNode = None,
                 end_ptr: GNode = None,
                 range_len: GNode = None,
                 shift_len: GNode = None,
                 external_iter: GNode = None
                 ):
        self._exp_use = exp_use

        self._shift_len = shift_len

        if external_iter is None:
            self._loop_ptr = start_ptr.nop()
        else:
            self._loop_ptr = external_iter

        if end_ptr is None:
            self._end_ptr = range_len
        else:
            self._end_ptr = end_ptr

    def open(self):
        graph_ctx.start_use_block(self._exp_use)
        graph_ctx.stationary_code('do {{')
        return graph_ctx.bind_scope(self._loop_ptr)

    def close(self):
        it = graph_ctx.bind_scope(self._loop_ptr)
        graph_ctx.stationary_code('{} = {};', it, it + self._shift_len)
        graph_ctx.stationary_code('}} while ({} < {});', it, self._end_ptr)
        graph_ctx.end_use_block()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
