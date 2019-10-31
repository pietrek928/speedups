from func import Func, func_reg
from loop import Loop
from proc_ctx import proc
from proc_descr import ProcDescr
from vtypes import v4f, float_

pd = ProcDescr(
    name='testproc',
    mem_levels=(
        ('regs', 16, 0, 0.0),
        ('L1', 3200, 1, 7.0)
    ),
    ops=(
        ('loadXv4f', v4f, 7.0, (1, 2), True),
        ('storeYv4f', None, 7.0, (6, 1), True),
        ('loadYfloat', float_, 7.0, (6, 1), True),
        ('storeYfloat', None, 7.0, (6, 1), True),
        ('nopYfloat', float_, 0.0, (6, 1), True),
        ('constYfloat', float_, 2.0, (3,), True),
        ('mulYfloatXfloat', float_, 5.5, (5,), True),
        ('addYfloatXfloat', float_, 5.5, (4,), True),
        ('loadYv4f', v4f, 7.0, (3,), True),
        ('addYv4fXv4f', v4f, 3.5, (3, 4), True),
        ('subYv4fXv4f', v4f, 3.0, (3, 4), True),
        ('mulYv4fXv4f', v4f, 5.5, (5,), True),
        ('negYv4', v4f, 5.0, (5,), True),
        ('notbitYv4f', v4f, 5.0, (5,), True),
        ('cvtYv4fXfloat', float_, 5.0, (7,), True)
    )
)


@Func(yo=1.0, elo=2.0)
def ttest():
    g = float_.load('ooooooo')
    a = float_.var('yo', const=False)
    b = float_.var('elo', const=True)
    c = float_.var('eloo', const=True, default=0.8)
    with Loop(
        start_ptr=float_.load('start'),
        end_ptr=float_.load('end'),
        shift_len=float_.load('shift')
    ) as it:
        with Loop(
                start_ptr=it,
                end_ptr=float_.load('end2'),
                shift_len=float_.load('shift2')
        ) as it2:
            it2.store('&d')
            (a * b * c * g).store('&c')


with proc(pd):
    ttest(elo=3.0, eloo=9.0)
    ttest.gen(dict(elo=5.0))

    print(func_reg)

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
