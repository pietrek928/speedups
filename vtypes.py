import threading
from collections import namedtuple

ctx = threading.local()


class VType:
    name = '?'
    x = 1
    y = 1

    def __str__(self):
        return self.name
    __repr__ = __str__


class float__(VType):
    name = 'float'


class v4f_(VType):
    name = 'v4f'
    x = 4
    y = 1


class v4x2f_(VType):
    name = 'v4x2f'
    x = 4
    y = 2


float_ = float__()
v4f = v4f_()
v4x2f = v4x2f_()

# class NodeType:
#     def __init__(self, name):
#         self.name = name
#
#     def __call__(self, *args, **kwargs):
#         ctx.p.

Dimension = namedtuple('n', 'size')


class ArrayDescr:
    def __init__(self, *dims, t: VType):
        self._dims = dims
        self._t = t

        if self._t.x != self._dims[0].size:
            raise ValueError('Invalid vector size')

    def _gen_pos(self, dims):
        pass

    def load(self, dims):
        sz = 1
        it = 0
        while it < len(dims) and self._dims[it] == dims[it]:
            sz *= dims[it].size
            it += 1
        if it < len(dims):
            pass
        #

