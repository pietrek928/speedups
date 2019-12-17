from collections import defaultdict
from itertools import chain, product
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Tuple, Union

from .utils import addi, muli
from .vtypes import Ptr, VType, int32_

Dimension = NamedTuple('Dimension', (('n', str), ('size', int)))


class CumDims(defaultdict):
    def __init__(self, *a, **k):
        super().__init__(lambda: 1, *a, **k)

    @staticmethod
    def from_dims(dims: Iterable[Dimension]) -> 'CumDims':
        cum_dims = CumDims()
        for dim in dims:
            cum_dims[dim.n] *= dim.size
        return cum_dims

    @staticmethod
    def from_cfg(dims: Iterable[Tuple]) -> 'CumDims':
        cum_dims = CumDims()
        for dim in dims:
            dim = Dimension(*dim)
            cum_dims[dim.n] *= dim.size
        return cum_dims

    def __mul__(self, d: 'CumDims'):
        return CumDims(
            (k, self[k] * d[k])
            for k in set(chain(self.keys(), d.keys()))
        )

    def __truediv__(self, d: 'CumDims'):
        rd = {}
        for k in set(chain(self.keys(), d.keys())):
            if self[k] % d[k]:
                raise ValueError(f'Cannot divide dimensions {self} / {d}')
            rd[k] = self[k] // d[k]
        return CumDims(rd)

    def __sub__(self, d: 'CumDims'):
        return CumDims(
            (k, self[k] - d[k])
            for k in set(chain(self.keys(), d.keys()))
            if self[k] < d[k]
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

    @property
    def indexes(self) -> Iterable['CumPos']:
        dnames, dsizes = zip(*self.items())
        for pos in product(*(range(i) for i in dsizes)):
            yield CumPos(zip(dnames, pos))

    @property
    def indexes_array(self) -> 'IndexesArray':
        return IndexesArray(
            self.items(),
            self.indexes
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


class OrdDims(Tuple[Dimension]):
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

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(
            dim.size for dim in self
        )

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(
            dim.n for dim in self
        )

    def __truediv__(self, d: Union['CumDims', 'OrdDims']):
        if isinstance(d, OrdDims):
            d = CumDims.from_dims(d)

        r = []
        for dim in self:
            s = dim.size
            if d[dim.n] > 1:
                if d[dim.n] > s:
                    assert d[dim.n] % s == 0
                    d[dim.n] //= s
                    s = 1
                else:
                    assert s % d[dim.n] == 0
                    s //= d[dim.n]
                    d[dim.n] = 1
            if s > 1:
                r.append(Dimension(dim.n, s))

        return OrdDims(r)


class ArrayDescr:
    def __init__(self, t: VType, dims: Iterable[Dimension]):
        self._t = t

        # vdims = []
        # dims = list(dims)
        # for i, sz in enumerate(t.dims):
        #     dim = dims[0]
        #     vdims.append(Dimension(n=dim.n, size=sz))
        #
        #     if dim.size == sz:
        #         del dims[0]
        #     if not (i == len(t.dims) - 1 and dim.size % sz == 0):
        #         raise ValueError(f'Dimension {dim.n} size invalid')
        #     dims[0] = Dimension(n=dim.n, size=dim.size // sz)

        self.dims: OrdDims = OrdDims(dims)
        # self._vdims: OrdDims = OrdDims(vdims)

    @property
    def size(self) -> int:
        n = 1
        for dim in self.dims:
            n = muli(n, dim.size)
        return n

    # def _vec_dims(self, dims: OrdDims) -> OrdDims:
    #     dims = tuple(dims)
    #     assert dims[:len(self._vdims)] == self._vdims
    #     return OrdDims(dims[len(self._vdims):])
    #
    # def _vec_ddims(self, ddims: CumDims) -> CumDims:
    #     ddims = ddims.copy()
    #     for vdim in self._vdims:
    #         if vdim.n in ddims:
    #             if ddims[vdim.n] % vdim.size:
    #                 raise ValueError('Dimension {} size not multiple of vector size'.format(vdim.n))
    #             ddims[vdim.n] //= vdim.size
    #         else:
    #             raise ValueError('Lacking dimension {}'.format(vdim.n))
    #     return ddims

    # @property
    # def dims(self) -> OrdDims:
    #     return self._dims

    @property
    def ddims(self) -> CumDims:
        return CumDims.from_dims(self.dims)

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

    # def indexes(self, ddims: CumDims, start=0):
    #     return self.block_indexes(self._vec_ddims(ddims), start)

    def block_indexes(self, block_ddims: Mapping[str, int], start=0) -> 'IndexesArray':
        ddims = CumDims(block_ddims)

        if isinstance(start, int):
            start_pos = start
        else:
            start_pos = 0

        items = IndexesArray(
            dims=self.dims,
            items=(start_pos,)
        )
        shift = 1
        for dim in self.dims:
            if ddims[dim.n] > 1:
                cur_size = min(ddims[dim.n], dim.size)
                if min(ddims[dim.n], dim.size) % cur_size:
                    raise ValueError('Dimension {} is not multiple, and is not taken as last')
                ddims[dim.n] //= cur_size
                items = items.extend_indexes(shift, Dimension(n=dim.n, size=cur_size))
            shift = muli(shift, dim.size)

        if start and not start_pos:
            items = items.map(lambda i: addi(start, i))

        return items

    def get_idx(self, pos: CumPos):
        return self.dims.get_idx(pos)


class StoredArray(ArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], arr):
        super().__init__(t, dims)
        self._arr = arr

    @property
    def arr_node(self):
        return Ptr(self._t)(self._arr)

    def load(self, ddims: Mapping[str, int], start=0):
        return self.block_indexes(ddims, start).map(
            lambda idx: self._t.load(self.arr_node, idx)
        )

    def store(self, a: 'MemArray', start=0):
        idxs = self.block_indexes(a.ddims, start)
        a = a.reshape(idxs.dims)
        a.map2(
            idxs,
            lambda v, idx: v.store(self.arr_node, idx)
        )


class MemArray(ArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], items: Iterable = None):
        super().__init__(t, dims)

        if items is not None:
            self._items = list(items)
        else:
            self._items = [None, ] * self.size
        assert self.size == len(self._items)

    def dim_size(self, dim_n: str) -> int:
        n = 1
        for dim in self.dims:
            if dim.n == dim_n:
                n *= dim.size
        return n

    @property
    def ddims_indexes(self) -> Iterable[CumPos]:
        return CumDims.from_dims(self.dims).indexes

    def map(self, func: Callable[[Any], Any]) -> 'MemArray':
        return MemArray(
            t=self._t,
            dims=self.dims,
            items=(
                func(item) for item in self._items
            )
        )

    def map2(self, a: 'MemArray', func: Callable[[Any, Any], Any]) -> 'MemArray':
        assert self.dims == a.dims
        return MemArray(
            t=self._t,
            dims=self.dims,
            items=(
                func(item1, item2) for item1, item2 in zip(self._items, a._items)
            )
        )

    def reshape(self, dims: Iterable[Dimension]) -> 'MemArray':
        r = MemArray(
            t=self._t,
            dims=dims
        )

        assert self.ddims == r.ddims

        for p in self.ddims_indexes:
            r.set(p, self.get(p))

        return r

    def block_iterate(self, block_ddims: CumDims) -> Iterable['MemArray']:  # TODO: shift order ?
        block_items = block_ddims.indexes_array
        for block_start in (self.ddims - block_ddims).indexes:
            yield block_items.shift(block_start).map(self.get)

    def get(self, pos: CumPos):
        return self._items[self.dims.get_idx(pos)]

    def set(self, pos: CumPos, val):
        self._items[self.dims.get_idx(pos)] = val

    # def __getitem__(self, *items):
    #     return self.get(items)


class IndexesArray(MemArray):
    def __init__(self, dims: Iterable[Dimension], items: Iterable):
        super().__init__(int32_, dims, items)

    def extend_indexes(self, shift, dim: Dimension) -> 'MemArray':
        shifted = [self._items]
        cur_shift = shift
        for i in range(1, dim.size):
            shifted.append(
                addi(v, shift) for v in self._items
            )
            cur_shift = addi(cur_shift, shift)
        return IndexesArray(
            dims=chain(self.dims, (dim,)),
            items=chain(*shifted)
        )

    def shift(self, p: CumPos):
        return self.map(lambda pp: pp + p)
