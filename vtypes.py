import threading
from collections import namedtuple, defaultdict
from itertools import chain, product
from typing import Iterable, Tuple, Mapping, NamedTuple, Dict, Callable, Any, List

from utils import muli, addi

ctx = threading.local()

Dimension = namedtuple('Dimension', ('n', 'size'))


class VType:
    name: str = '?'
    dims: Tuple[int] = ()

    def format(self, v):
        return str(v)

    def __eq__(self, v):
        return self.name == v.name

    def __str__(self):
        return self.name

    __repr__ = __str__


OpDescr = NamedTuple('op_descr', (('name', str), ('op_id', int), ('ordered', int), ('out_t', VType)))


class bool__(VType):
    name = 'bool'

    def format(self, v):
        return bool(v)


class int32__(VType):
    name = 'int32'

    def format(self, v):  # TODO: check range
        return int(v)


class float__(VType):
    name = 'float'

    def format(self, v):
        return float(v)


class v4f_(VType):
    name = 'v4f'
    dims = (4,)


class v4x2f_(VType):
    name = 'v4x2f'
    dims = (4, 2)


bool_ = bool__()
int32_ = int32__()
float_ = float__()
v4f = v4f_()
v4x2f = v4x2f_()


# class NodeType:
#     def __init__(self, name):
#         self.name = name
#
#     def __call__(self, *args, **kwargs):
#         ctx.p.


class CumDims(defaultdict):
    def __init__(self, *a, **k):
        super().__init__(lambda: 1, *a, **k)

    @staticmethod
    def from_dims(dims: Iterable[Dimension]) -> 'CumDims':
        cum_dims = CumDims()
        for dim in dims:
            cum_dims[dim.n] *= dim.size
        return cum_dims

    def __mul__(self, d: 'CumDims'):
        return CumPos(
            (k, self[k] * d[k])
            for k in set(chain(self.keys(), d.keys()))
        )

    # def to_pos(self):
    #     return CumPos(**{
    #         k: v for k, v in self.items()
    #         if v > 1
    #     })

    @property
    def size(self) -> int:
        n = 1
        for size in self.values():
            n = muli(n, size)
        return n

    def poss(self) -> Iterable['CumPos']:
        dnames, dsizes = zip(*self.items())
        for pos in product(*(range(i) for i in dsizes)):
            yield CumPos(zip(dnames, pos))

    def poss_array(self) -> Tuple['OrdDims', Tuple['CumPos']]:
        dims = OrdDims(*self.items())
        dnames, dsizes = zip(*dims)

        return dims, tuple(
            CumPos(zip(dnames, pos))
            for pos in product(*(range(i) for i in dsizes))
        )


class CumPos(defaultdict):
    def __init__(self, *a, **k):
        super().__init__(lambda: 0, *a, **k)

    def copy(self):
        return CumPos(self)

    def __add__(self, d: 'CumPos') -> 'CumPos':
        return CumPos(
            (k, self[k] + d[k])
            for k in set(chain(self.keys(), d.keys()))
        )

    def __sub__(self, d: 'CumPos') -> 'CumPos':
        return CumPos(
            (k, self[k] - d[k])
            for k in set(chain(self.keys(), d.keys()))
        )

    def shift(self, poss: Iterable['CumPos']) -> Iterable['CumPos']:
        for p in poss:
            yield self + p


class OrdDims(tuple):
    def __init__(self, dims: Iterable[Dimension]):
        super().__init__(dims)

    def get_idx(self, pos: CumPos):
        pos = pos.copy()

        idx = 0
        shift = 1

        for dim in self:
            if not pos[dim.n]:
                continue
            if pos[dim.n] > 0:
                v = pos[dim.n] % dim.size
                pos[dim.n] //= dim.size
            else:
                v = -((-pos[dim.n]) % dim.size)
                pos[dim.n] = -((-pos[dim.n]) // dim.size)

            idx += v * shift
            shift *= dim.size

        return idx


def _aggr_dims(dims: Iterable[Dimension]) -> Dict[str, int]:
    ddims = defaultdict(lambda: 1)
    for dim in dims:
        ddims[dim.n] *= dim.size
    return ddims


def _ddims_poss(ddims: Mapping[str, int]) -> Iterable[Mapping[str, int]]:  # !!!
    dnames, dsizes = zip(*ddims.items())
    for pos in product(*(range(i) for i in dsizes)):
        yield zip(dnames, pos)


class ArrayDescr:
    def __init__(self, t: VType, dims: Iterable[Dimension]):
        self._t = t

        vdims = []
        dims = list(dims)
        for i, sz in enumerate(t.dims):
            dim = dims[0]
            vdims.append(Dimension(n=dim.n, size=sz))

            if dim.size == sz:
                del dims[0]
            if not (i == len(t.dims) - 1 and dim.size % sz == 0):
                raise ValueError('Dimension {} size invalid'.format(dim.n))
            dims[0] = Dimension(n=dim.n, size=dim.size // sz)

        self._dims: OrdDims = OrdDims(dims)
        self._vdims: OrdDims = OrdDims(vdims)

    @property
    def size(self) -> int:
        n = 1
        for dim in self._dims:
            n = muli(n, dim.size)
        return n

    def _vec_dims(self, dims: OrdDims) -> OrdDims:
        dims = tuple(dims)
        assert dims[:len(self._vdims)] == self._vdims
        return OrdDims(dims[len(self._vdims):])

    def _vec_ddims(self, ddims: CumDims) -> CumDims:
        ddims = ddims.copy()
        for vdim in self._vdims:
            if vdim.n in ddims:
                if ddims[vdim.n] % vdim.size:
                    raise ValueError('Dimension {} size not multiple of vector size'.format(vdim.n))
                ddims[vdim.n] //= vdim.size
            else:
                raise ValueError('Lacking dimension {}'.format(vdim.n))
        return ddims

    @property
    def dims(self) -> OrdDims:
        return self._vdims + self._dims

    @property
    def ddims(self) -> CumDims:
        return CumDims.from_dims(self.dims)

    @property
    def ddims_vec_poss(self) -> Iterable[CumPos]:
        return CumDims.from_dims(self._dims).poss()

    # def _get_idx_vec(self, pos: CumDims):
    #     shift = 1
    #     pos = defaultdict(lambda: 0, pos)
    #     idx = 0
    #
    #     for dim in reversed(self._dims):
    #         if pos[dim.n] <= 0:
    #             continue
    #         v = pos[dim.n] % dim.size
    #         pos[dim.n] //= dim.size
    #
    #         idx += v * shift
    #         shift *= dim.size
    #     return idx

    def indexes(self, ddims: CumDims, start=0):
        return self.vec_indexes(self._vec_ddims(ddims), start)

    def vec_indexes(self, vec_ddims: Mapping[str, int], start=0):
        ddims = defaultdict(
            lambda: 1,
            vec_ddims
        )

        if isinstance(start, int):
            start_idx = start
        else:
            start_idx = 0

        items = RegArray(
            t=self._t,
            dims=self._vdims,
            items=(start_idx,)
        )
        shift = 1
        for dim in self._dims:
            if ddims[dim.n] > 1:
                cur_size = min(ddims[dim.n], dim.size)
                if min(ddims[dim.n], dim.size) % cur_size:
                    raise ValueError('Dimension {} is not multiple, and is not taken as last')
                ddims[dim.n] //= cur_size
                items = items.extend(shift, Dimension(n=dim.n, size=cur_size))
            shift = muli(shift, dim.size)

        if start and not start_idx:
            items = items.map(lambda i: addi(start, i))

        return items

    def get_idx(self, vec_pos: CumDims):
        return self.dims.get_idx(vec_pos)


class MemArray(ArrayDescr):
    def load(self, p, dims: Mapping[str, int], start=0):
        return self.vec_indexes(dims, start).map(
            lambda i: p.load(v4f, i)
        )

    def store(self, a: 'RegArray', start=0):
        idxs = self.indexes(a.ddims, start)
        a = a.reshape(idxs.dims)
        # !!!!!!!!


class RegArray(ArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], items: Iterable = None):
        super().__init__(t, dims)

        if items is not None:
            self._items = list(items)
        else:
            self._items = [None, ] * self.size
        assert self.size == len(self._items)

    def dim_size(self, dim_n: str) -> int:
        n = 1
        for dim in self._dims:
            if dim.n == dim_n:
                n *= dim.size
        return n

    def map(self, func: Callable[[Any], Any]) -> 'RegArray':
        return RegArray(
            t=self._t,
            dims=self.dims,
            items=(
                func(item) for item in self._items
            )
        )

    def map2(self, a: 'RegArray', func: Callable[[Any, Any], Any]) -> 'RegArray':
        assert self.dims == a.dims
        return RegArray(
            t=self._t,
            dims=self.dims,
            items=(
                func(item1, item2) for item1, item2 in zip(self._items, a._items)
            )
        )

    def extend(self, shift, dim: Dimension) -> 'RegArray':
        shifted = [self._items]
        cur_shift = shift
        for i in range(1, dim.n):
            shifted.append(
                addi(v, shift) for v in self._items
            )
            cur_shift = addi(cur_shift, shift)
        return RegArray(
            t=self._t,
            dims=chain(self.dims, (dim,)),
            items=chain(*shifted)
        )

    def reshape(self, dims: Iterable[Dimension]) -> 'RegArray':
        r = RegArray(
            t=self._t,
            dims=dims
        )

        # TODO: reshape vectors
        assert self._vdims == r._vdims
        assert self.ddims == r.ddims

        for p in self.ddims_vec_poss:
            r.set_vec(p, self.get_vec(p))

        return r

    def block_iterate(self, block_dims: CumDims):
        block_items = block_dims.poss()
        for block_start in (self.ddims - block_dims).poss():  # !!!!!!!!
            vdims, poss = block_start.poss_array()
            a = RegArray(
                t=self._t,
                dims=self._t.dims + vdims,
                items=block_start.shift(block_items)
            )
            return a.map(lambda v: self.get_vec(v))

    def get_vec(self, pos: CumPos):
        return self._items[self._dims.get_idx(pos)]

    def set_vec(self, pos: CumPos, val):
        self._items[self._dims.get_idx(pos)] = val

    # def __getitem__(self, *items):
    #     return self.get(items)


class ArrayIteration:
    def __init__(self, dims: Iterable[Dimension]):
        self._dims: List[Dimension] = list(dims)
        self._orig_dims: Tuple[Dimension] = tuple(self._dims)

    def get_dim(self, n):
        for d in self._dims:
            if d.n == n and d.size > 1:
                return d.size

    def move_dim(self, d: Dimension):
        shift = 1

        for i, di in enumerate(self._dims):
            if di.n == d.n:
                if di.size <= d.size:
                    self._dims[i] = Dimension(
                        n=di.n,
                        size=di.size // d.size
                    )
                break
            shift *= self._orig_dims[i].size

        return shift, shift * d.size
