from typing import Iterable, List, Tuple, Mapping

from .array import ArrayDescr, CumDims, Dimension, MemArray, OrdDims, StoredArray
from .graphval import GraphVal
from .utils import get_p2
from .vtypes import VType


class VecArrayDescr(ArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], vdim_names: Iterable[str]):
        super().__init__(t, dims)
        self._vdim_names = tuple(vdim_names)

    @property
    def vec_type_dims(self) -> OrdDims:
        return OrdDims(
            Dimension(n, s)
            for n, s in zip(self._vdim_names, self._t.dims)
        )

    @property
    def vdims(self) -> OrdDims:
        return OrdDims(
            self.vec_type_dims + self.dims
        )

    @property
    def vddims(self) -> CumDims:
        return CumDims.from_dims(self.vdims)

    def _split_vdims(self, vdims: Iterable[Dimension]) -> Tuple[OrdDims, OrdDims]:
        vdims = OrdDims(vdims)
        assert self.vddims == CumDims.from_dims(vdims)

        tdims = self.vec_type_dims
        n = len(tdims)
        if not vdims.shape[:len(tdims.shape)] == tdims.shape:
            raise ValueError(f'Specified invalid dimensions `{vdims}` for array with vector dim {tdims}')

        return OrdDims(vdims[:n]), OrdDims(vdims[n:])


class VecMemArray(MemArray, VecArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], vdim_names: Iterable[str], items: Iterable = None):
        super().__init__(t, dims, items)
        self._vdim_names = tuple(vdim_names)

    @classmethod
    def from_mem_array(cls, mem_array: MemArray, vdim_names: Iterable[str]):
        return cls(
            t=mem_array._t,
            dims=mem_array.dims,
            vdim_names=vdim_names,
            items=mem_array._items
        )

    def _shuf_tdim(self, n: int, ndim: Dimension) -> 'VecMemArray':
        ntdims = list(self.vec_type_dims)
        assert n < len(ntdims)

        odim = ntdims[n]
        ntdims[n] = ndim
        if odim == ndim:
            return self
        assert odim.size == ndim.size

        r = self.reshape(odim + self.dims / OrdDims((odim, )))

        vecs: List[GraphVal, ...] = list(r._items)
        assert vecs

        dim_sz = odim.size
        dim_shuf = tuple(int(i == n) for i in range(len(ntdims)))
        assert len(vecs) % dim_sz == 0

        for it_n in reversed(range(get_p2(dim_sz))):
            s = 2 ** it_n
            for i in range(0, dim_sz, s * 2):
                for j in range(s):
                    vecs[i + j], vecs[i + j + s] = vecs[i + j].shuf(vecs[i + j + s], *dim_shuf)

        return VecMemArray(
            t=r._t,
            dims=r.dims,
            vdim_names=OrdDims(ntdims).names,
            items=vecs
        )

    def tdims_reshape(self, ntdims: Iterable[Dimension]) -> 'VecMemArray':
        ntdims = OrdDims(ntdims)
        assert ntdims.shape == self._t.dims

        r = self
        for i, dim in enumerate(ntdims):
            r = r._shuf_tdim(i, dim)
        return r

    def vec_reshape(self, vdims: Iterable[Dimension]) -> 'VecMemArray':
        tdims, dims = self._split_vdims(vdims)
        return self.tdims_reshape(tdims).reshape(dims)


class VecStoredArray(StoredArray, VecArrayDescr):
    def __init__(self, t: VType, dims: Iterable[Dimension], vdim_names: Iterable[str], arr):
        super().__init__(t, dims, arr)
        self._vdim_names = tuple(vdim_names)

    def load(self, vddims: Mapping[str, int], start=0):
        vddims = CumDims(vddims)
        ddims = vddims / CumDims.from_dims(self.vec_type_dims)
        return VecMemArray.from_mem_array(
            mem_array=super().load(ddims, start),
            vdim_names=self._vdim_names
        )

    def store(self, a: 'VecMemArray', start=0):
        a = a.tdims_reshape(self.vec_type_dims)
        super().store(a, start)
