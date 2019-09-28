from typing import Iterable, List, Tuple, Any, Dict

import optim

from gnode import GNode, OpScope


class GraphOptim:
    def __init__(self, p, op_l: Iterable[GNode], op_scopes: Iterable[OpScope]):
        self.p = p
        self.op_l: Tuple[GNode] = tuple(op_l)
        self._prog = optim._prog(p, self.op_nums, self._simple_graph(), op_scopes)

    @property
    def op_nums(self):
        return tuple(
            v.op.op_id for v in self.op_l
        )

    def _simple_graph(self, ord: Iterable[int] = None) -> List[Tuple[int]]:
        if ord is None:
            ord = range(len(self.op_l))

        G = []
        nums: Dict[Any, int] = {}
        for i in ord:
            v = self.op_l[i]
            nums[v.orig] = i
            G.append(tuple(
                nums[vn.orig] for vn in v.a
            ))

        return G
