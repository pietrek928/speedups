from typing import Iterable, Tuple

from speedutils.graphval import GraphVal
from speedutils.proc_ctx import graph_ctx
from speedutils.vtypes import float_


class Float(float_):
    def inv(self):
        return self.gen_one() / self

    def invsqrt(self):
        return self.from_op('inversesqrt', self)

    @classmethod
    def from_noise(cls, ):
        return cls.from_op('noise1')

    def iszero(self):
        return self._expr == 0.

    def isone(self):
        return self._expr == 1.

    def isminusone(self):
        return self._expr == -1.


class Half(Float):
    typename = 'half'


def slice2range(s: slice, array_len) -> range:
    step = s.step
    if not step:
        step = 1
    start = s.start
    if start is None:
        start = 0 if step > 0 else array_len - 1
    stop = s.stop
    if stop is None:
        stop = array_len if step > 0 else -1

    return range(start, stop, step)


class Vec(GraphVal):
    type_name = 'vec'
    ELEM_NAMES = 'xyzw'

    @classmethod
    def from_noise(cls, ):
        return cls.from_op(f'noise{cls.len}')

    # @classmethod
    # def parse_expr(cls, expr):
    #     expr = convert_args(expr)
    #     assert len(expr) == cls.len
    #     return expr

    @property
    def isaddsub(self):
        return self.isconst and all(
            v.zero or v.one or v.minusone
            for v in self._expr
        )

    @classmethod
    def from_(cls, v):
        if v.dims[0] == cls.dims[0]:
            return v
        elif v.dims[0] < cls.dims[0]:
            zeros = (Float.gen_zero(),) * (cls.dims[0] - v.dims[0])
            return cls.concat(v, *zeros)
        else:
            return v[:cls.dims[0]]

    def item(self, n) -> GraphVal:
        if n < 0:
            n += self.dims[0]
        if not 0 <= n < self.dims[0]:
            raise IndexError(f'Invalid index {n} for {self.type_name}')
        if self.const:
            return convert_arg(self._expr[n])
        return self.from_expr(
            f'{{}}.{self.ELEM_NAMES[n]}', self,
            op=graph_ctx.find_spec_op('get_elem', type(self))
        )

    def __getitem__(self, pos):
        if isinstance(pos, int):
            return self.item(pos)
        elif isinstance(pos, slice):
            pos_range = slice2range(pos, self.dims[0])
            if pos_range.start == 0 and pos_range.step == 1:
                t = VEC_BY_LEN[pos_range.stop]
                return self.from_expr(
                    f'{{}}.{self.ELEM_NAMES[:pos_range.stop]}', self,
                    op=graph_ctx.find_spec_op('concat', t)
                )
            return self.concat(
                *map(self.item, pos_range)
            )
        elif isinstance(pos, tuple):
            return self.concat(*(
                self.item(v) for v in pos
            ))

    @staticmethod
    def concat(*args):
        args = convert_args(args)
        if len(args) == 1:
            return args[0]

        out_len = sum(map(lambda v: v.dims[0], args))
        out_type = VEC_BY_LEN.get(out_len)
        if out_type is None:
            raise ValueError(f'Invalid vector length {out_len}')

        return out_type.from_spec_op(
            'concat', *args
        )

    # @classmethod
    # def format_const(cls, v):
    #     vv = ', '.join(map(str, v))
    #     return f'{cls.typename}({vv})'

    def dp(self, v):
        if self.neg:
            return -(v * (-self))
        # if self.get('invdiv'):
        #     return (v * self.setnot('invdiv')).setnot('invdiv')
        if self.zero or v.one:
            return self
        if self.one or v.zero:
            return v

        # assert isinstance(v, Vec)
        # w = self
        # if w.type.len < v.type.len:
        #     v = v[:w.type.len]
        # elif w.type.len > v.type.len:
        #     w = w[:v.type.len]
        # if not v.istype(self.type):
        #     raise ValueError(f'Cannot do `dp` with {self.typename} and {v.typename}')

        # if self.isaddsub or v.isaddsub:
        #

        return self.from_op('dp', self, v)

    # def __mul__(self, v):
    #     v = convert_arg(v)
    #     rt = None
    #     if v.istype(Float):
    #         rt = self.type
    #     return self._call_op('*', v, rt=rt)
    #
    # def __truediv__(self, v):
    #     v = convert_arg(v)
    #     rt = None
    #     if v.istype(Float):
    #         rt = self.type
    #     return self._call_op('/', v, rt=rt)

    def qdist(self):
        return self.dp(self)

    def dist(self):
        return self.from_op('length', self)

    def vers(self):
        return self.from_op('normalize', self)


class Vec2(Vec):
    type_name = 'vec2'
    dims = (2,)


class Vec3(Vec):
    type_name = 'vec3'
    dims = (3,)


class Vec4(Vec):
    type_name = 'vec4'
    dims = (4,)


class Mat(GraphVal):
    type_name = 'mat'

    # @classmethod
    # def from_(cls, *a):
    #     if not a:
    #         a = (1.,)
    #     a = convert_args(a)
    #     if len(a) == 1:
    #         assert a[0].istype(Float, Mat)
    #         return get_context_shader().add_function_call(cls, cls.typename, a[0])
    #
    #     raise ValueError(f'Matrix creation from {a} unsupported')

    def item(self, n):
        if n < 0:
            n += self.dims[0]
        if not 0 <= n < self.dims[0]:
            raise IndexError(f'Invalid index {n} for {self.typename}')
        rt = VEC_BY_LEN[self.dims[0]]
        return rt.from_expr(f'{{}}[{n}]', self)

    def inv(self):
        return self.from_op('invert', self)

    def __mul__(self, v: GraphVal):
        if v.istype(Mat):
            return super().__mul__(v)
        elif v.istype(Vec):
            assert self.dims[1] == v.dims[0]
            return super().__mul__(v)
        else:
            raise ValueError(f'Unsupported multiplication of {self} and {v}')


class Mat2(Mat):
    type_name = 'mat2'
    dims = (2,) * 2


class Mat3(Mat):
    type_name = 'mat3'
    dims = (3,) * 2


class Mat4(Mat):
    type_name = 'mat4'
    dims = (4,) * 2


VEC_BY_LEN = {
    1: Float,
    2: Vec2,
    3: Vec3,
    4: Vec4
}


def convert_arg(v) -> GraphVal:
    if isinstance(v, GraphVal):
        return v
    elif isinstance(v, float):
        return Float.from_const(v)
    elif isinstance(v, (tuple, list)):
        if len(v) == 1:
            return Float(v[0])
        elif len(v) in VEC_BY_LEN:
            return VEC_BY_LEN[len(v)](v)
    raise ValueError(f'Count not convert {type(v)} {v}')


def convert_args(arg_list: Iterable) -> Tuple[GraphVal, ...]:
    return tuple(
        map(convert_arg, arg_list)
    )


class Sampler2D(GraphVal):
    typename = 'sampler2D'
    components = 3

    # TODO: type opts ?
    # def __init__(self, expr: str, components=3):
    #     super().__init__(expr)
    #     self._rt = VEC_BY_LEN.get(components)
    #     if self._rt is None:
    #         raise ValueError(f'Wrong number of components {components}')

    def sample(self, coords: Vec2):
        # return self._rt.from_(
        #     get_context_shader().add_function_call(
        #         Vec4, 'texture', self, coords
        #     )
        # )
        return self.from_op('texture', self, coords)
