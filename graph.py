from optim import _prog


class ProgGraph:
    def __init__(self, p, op_l):
        self.p = p
        self.op_l = op_l
        self._prog = _prog(p, self.op_nums, self._simple_graph())

    @property
    def op_nums(self):
        return tuple(
            v.op.op_id for v in self.op_l
        )

    def _simple_graph(self, ord=None):
        if ord is None:
            ord = range(len(self.op_l))

        G = []
        nums = {}
        for i in ord:
            v = self.op_l[i]
            nums[v.orig] = i
            G.append(tuple(
                nums[vn.orig] for vn in v.a
            ))

        return G

    def gen_code(self, ord=None):
        if ord is None:
            ord = range(len(self.op_l))

        nums = {}
        for i in ord:
            v = self.op_l[i]
            nums[v.orig] = i
            v.print_op(nums)

