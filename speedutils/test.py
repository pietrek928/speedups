from speedutils.shader.base import SimpleFragmentShader, SimpleVertexShader

from .proc_ctx import new_graph, proc
from .proc_descr import CvtOp, FuncOp, Op, ProcDescr, SignOp, SimpleLoadOp, SimpleStoreOp, TypedNoargOp
from .shader.types import ConcatOp, Mat3, Mat4, Vec2, Vec3, Vec4
from .vtypes import float_, int32_

pd = ProcDescr(
    name='testproc',
    mem_levels=(
        ('regs', 16, 0, 0.0),
        ('L1', 3200, 1, 7.0)
    ),
    ops=(
        # ('loadXv4f', v4f, 7.0, (1, 2), True),
        # ('storeYv4f', None, 7.0, (6, 1), True),
        # ('loadYfloat', float_, 7.0, (6, 1), True),
        # ('storYfloat', None, 7.0, (6, 1), True),
        # ('nopYfloat', float_, 0.0, (6, 1), True),
        # ('constYfloat', float_, 2.0, (3,), True),
        # ('mulYfloatXfloat', float_, 5.5, (5,), True),
        # ('addYfloatXfloat', float_, 5.5, (4,), True),
        # ('loadYv4f', v4f, 7.0, (3,), True),
        # ('addYv4fXv4f', v4f, 3.5, (3, 4), True),
        # ('subYv4fXv4f', v4f, 3.0, (3, 4), False),
        # ('mulYv4fXv4f', v4f, 5.5, (5,), True),
        # ('negYv4', v4f, 5.0, (5,), True),
        # ('notbitYv4f', v4f, 5.0, (5,), True),
        # ('cvtYv4fXfloat', float_, 5.0, (7,), True),
        # ('cvtYfloatXint32', int32_, 5.0, (7,), True),
        #
        # ('zeroYint32Y', int32_, 1.0, (4,), True),
        # ('zeroYfloat', float_, 1.0, (4,), True),
        # ('constYint32', int32_, 1.0, (4,), True),
        # ('loadYint32', int32_, 7.0, (6, 1), True),
        # ('storYint32', None, 7.0, (6, 1), True),
        # ('nopYint32', int32_, 0.0, (4,), True),
        # ('addYint32Xint32', int32_, 1.0, (4,), True),
        # ('mulYint32Xint32', int32_, 3.0, (4,), True),
        # ('mulYint32Xfloat', float_, 3.0, (4,), True),
        # ('addYint32Xfloat', float_, 3.0, (4,), True),
        #
        # ('concatYvec4', Vec4, 1.0, (3,), True),
        # ('concatYvec2', Vec2, 1.0, (3,), True),
        # ('get_elemYvec4', float_, 1.0, (3,), True),
        # ('cvtYmat4Xmat3', Mat3, 4.0, (3,), True),
        # ('mulYmat4Xvec4', Vec4, 4.0, (4,), True),
        # ('mulYmat3Xvec3', Vec3, 4.0, (4,), True),
        # ('normalizeYvec3', Vec3, 5.0, (4,), True),
        # ('divYvec2Xfloat', Vec2, 4.0, (4,), True),
        # ('loadYmat4', Mat4, 8.0, (6, 1), True),
        # ('loadYvec4', Vec4, 4.0, (6, 1), True),
        # ('loadYvec3', Vec3, 4.0, (6, 1), True),
        # ('loadYvec2', Vec2, 4.0, (6, 1), True),
        # ('storYvec4', None, 4.0, (6, 1), True),
        # ('storYvec3', None, 4.0, (6, 1), True),
        # ('storYvec2', None, 4.0, (6, 1), True),
        SimpleLoadOp(ret_t=Mat4, exec_t=8.0, ports=(6, 1)),
        SimpleLoadOp(ret_t=Vec4, exec_t=4.0, ports=(6, 1)),
        SimpleLoadOp(ret_t=Vec3, exec_t=4.0, ports=(6, 1)),
        SimpleLoadOp(ret_t=Vec2, exec_t=4.0, ports=(6, 1)),
        SimpleStoreOp(Vec4, exec_t=1.0, ports=(3,)),
        SimpleStoreOp(Vec3, exec_t=1.0, ports=(3,)),
        SimpleStoreOp(Vec2, exec_t=1.0, ports=(3,)),
        SignOp('mul', '*', (Mat4, Vec4), ret_t=Vec4, exec_t=4.0, ports=(4,)),
        SignOp('mul', '*', (Mat3, Vec3), ret_t=Vec3, exec_t=3.0, ports=(4,)),
        SignOp('mul', '*', (Vec3, float_), ret_t=Vec3, exec_t=4.0, ports=(4,), args_ordered=False),
        SignOp('div', '/', (Vec2, float_), ret_t=Vec2, exec_t=4.0, ports=(4,)),
        FuncOp('normalize', args_t=(Vec3,), ret_t=Vec3, exec_t=3.0, ports=(4,)),
        FuncOp('dot', args_t=(Vec3, Vec3), ret_t=Vec3, exec_t=3.0, ports=(4,)),
        ConcatOp(Vec4, exec_t=1.0, ports=(3,)),
        ConcatOp(Vec2, exec_t=1.0, ports=(3,)),
        TypedNoargOp(name='zero', ret_t=float_, exec_t=1.0, ports=(4,)),
        TypedNoargOp(name='const', ret_t=float_, exec_t=1.0, ports=(3,)),
        TypedNoargOp(name='const', ret_t=Vec3, exec_t=1.0, ports=(3,)),
        Op(name='get_elemYvec3', ret_t=float_, exec_t=1.0, ports=(3,)),
        Op(name='get_elemYvec4', ret_t=float_, exec_t=1.0, ports=(3,)),
        CvtOp(in_t=Mat4, ret_t=Mat3, exec_t=1.0, ports=(3,)),
    )
)


# @LoopFunc(
#     loop_dims=(('y', 'y_sz'), ('z', 'z_sz'), ('x', 8), ('y', 8), ('z', 16)),
#     block_ddims=(('x', 2), ('y', 4), ('z', 8))
# )
# @Func(yo=1.0, elo=2.0)
# def ttest():
#     g = float_.load('ooooooo')
#     a = float_.var('yo', const=False)
#     b = float_.var('elo', const=True)
#     c = float_.var('eloo', const=True, default=0.8)
#     (a * b * c * g).store('&c')
# with Loop(
#     start_val=float_.load('start'),
#     end_val=float_.load('end'),
#     shift_len=float_.load('shift')
# ) as it:
#     with Loop(
#             start_val=it,
#             end_val=float_.load('end2'),
#             shift_len=float_.load('shift2')
#     ) as it2:
#         it2.store('&d')
#         (a * b * c * g).store('&c')

# @CLFunc(yo=1.0, elo=2.0)
# @GpuLoopFunc(CUDAFunc, loop_dims=(('y', 4), ('x', 2), ('y', 'y_sz'),), block_ddims=(('x', 2), ('y', 4)))
# def cuda_test(it_x, it_y):
#     (
#             it_x * it_y + func_ctx.get_var('elo', float_)
#     ).store('&aaa')
#
#
# with proc(pd):
#     # cuda_test.gen(dict(elo=2.5))
#     with new_graph():
#         # cuda_test(it_dims=(1, 2, 3, 4))
#         cuda_test(elo=3.5, y_sz=12345)
#         graph_ctx.gen_code()
#     # ttest(elo=3.0, eloo=9.0, y_sz=12345, z_sz=67899)
#     # ttest.gen(dict(elo=5.0))
#
#     for n, (f, opts) in func_reg.items():
#         f.gen({
#             n: t.format(v)
#             for n, t, v in opts
#         })
#     print(func_reg)

def test_graph():
    with proc(pd), new_graph() as g:
        a = float_.var('yooooo')
        b = float_.var('yeeeee')
        c = a + b
        c *= a
        c = c.cvt(int32_)
        c.store(int32_.var('arr xD'), int32_.var('pp'))

    g.print_graph()


# def test_func():
#     @Func(yo=1.0, elo=2.0)
#     def ttest():
#         a = float_.var('yo', const=False)
#         b = float_.var('elo', const=True)
#         c = float_.var('eloo', const=True, default=0.8)
#         (a * b * c * g).store('&c')

def test_shader():
    with proc(pd):
        vert_shader = SimpleVertexShader()
        vert_content = vert_shader.render()
        print(vert_content)
        # with open('a.vert', 'w') as f:
        #     f.write(vert_content)

        frag_shader = SimpleFragmentShader(
            inputs=vert_shader.output_params, uniform_inputs=vert_shader.uniform_params
        )
        frag_content = frag_shader.render()
        print(frag_content)

# test_graph()
test_shader()

# ttest()

# p = FlowGraph(mem_levels=mem_levels, ops=ops)
#
# t = []
# for i in range(100):
#     t.append(p.load(v4f, str(i)))
#
# for i in range(100):
#     t[i] += t[(i + 3) % 100]
# for i in range(100):
#     t[i] *= t[(i + 4) % 100]
# for i in range(100):
#     t[i] += t[(i + 5) % 100]
#
# for i in range(100):
#     p.store(t[i], str(i))
#
# # p.print_graph()
# # p.used_ordered().gen_code()
#
# import time
# start = time.time()
# test(p.used_ordered()._prog)
# end = time.time()
# print(end - start)
