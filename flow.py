from collections import defaultdict
from typing import Dict, Iterable, List

from optim import _proc

from gnode import OpNode, ConstNode, StoreNode, LoadNode, CvtNode, GNode, OpDescr
from graph import ProgGraph
from utils import str_list
from vtypes import VType


class ScopeDescr:
    def __init__(self, exp_use: float):
        self._exp_use = exp_use
        self._nodes = []

    def append(self, v: 'GNode'):
        self._nodes.append(v)


class FlowGraph:
    def __init__(self, mem_levels, ops):
        self.ops: Dict[str, OpDescr] = {}
        ports = set(sum(
            (o[3] for o in ops),
            tuple(m[2] for m in mem_levels)
        ))
        p = _proc(len(ports))
        port2n = dict(zip(*reversed(list(zip(*enumerate(ports))))))
        for n, size, port, load_time in mem_levels:
            p.new_mem_level(
                size, port2n[port], load_time
            )
        for n, out_t, len_t, ports, ordered in ops:
            op_id = p.new_op(
                len_t,
                [port2n[p] for p in ports]
            )
            self.ops[n] = OpDescr(n, op_id, ordered, out_t)
        self._op_idx: Dict[str, GNode] = {}
        self._proc: _proc = p
        self._scope_stack: List[ScopeDescr] = []

    def find_op(self, n: str, a: Iterable[GNode]):
        t = str_list(v.type for v in a)
        try:
            op_n = '{}Y{}'.format(n, 'X'.join(t))
            return self.ops[op_n]
        except KeyError:
            op_n = '{}Y{}'.format(n, 'X'.join(sorted(t)))
            op = self.ops[op_n]
            if not op.ordered:
                return op
            raise ValueError('Wrong argument order')

    def find_const_op(self, t: VType):
        op_n = 'constY{}'.format(t)
        return self.ops[op_n]

    def find_load_op(self, t: VType):
        op_n = 'loadY{}'.format(t)
        return self.ops[op_n]

    def find_cvt_op(self, v: GNode, t: VType):
        op_n = 'cvtY{}X{}'.format(v.type, t)
        return self.ops[op_n]

    def find_store_op(self, t: VType):
        op_n = 'storeY{}'.format(t)
        return self.ops[op_n]

    def _add_n(self, v: GNode):
        k = v.key
        if k in self._op_idx:
            return self._op_idx[k].copy()
        else:
            self._op_idx[k] = v
            self._scope_stack[v.scope_n].append(v)
            return v

    def new_scope(self, exp_use):
        self._scope_stack.append(
            ScopeDescr(
                exp_use=exp_use
            )
        )

    # def end_scope(self):
    #     self._scope_stack.pop()

    def get_scope_n(self, *a: GNode) -> int:
        scopes = tuple(
            v.scope_n for v in a
        )
        try:
            return max(scopes)
        except ValueError:
            return len(self._scope_stack) - 1

    def op(self, n, *a: GNode):
        return self._add_n(
            OpNode(self, n, a)
        )

    def cvt(self, v: GNode, t: VType):
        return self._add_n(
            CvtNode(self, v, t)
        )

    def load(self, t: VType, val):
        return self._add_n(
            LoadNode(self, t, val)
        )

    def store(self, v: GNode, val):
        return self._add_n(
            StoreNode(self, v, val)
        )

    def const(self, t: VType, v):
        # v = self._format_const(t, v)
        return self._add_n(
            ConstNode(self, t, v)
        )

    def zero(self, t: VType):
        return self._add_n(
            OpNode(self, 'zeroY{}'.format(t), ())
        )

    def select_used(self):
        stack = {}
        used = set()

        for v in self._op_idx.values():
            if v.type is None:
                stack[v.orig] = v

        used_items = []
        while stack:
            o, v = stack.popitem()
            used.add(o)

            used_items.append(v)
            for nv in v.a:
                if nv.orig not in used:
                    stack[nv.orig] = nv

        return used_items

    def used_ordered(self):
        g_rev = defaultdict(lambda: [])
        cnt = {}

        used = self.select_used()
        for v in used:
            cnt[v.orig] = len(v.a)
            for nv in v.a:
                g_rev[nv.orig].append(v)

        stack = []
        for v in used:
            if not cnt[v.orig]:
                stack.append(v)

        n = 1
        ordered = []
        while stack:
            v = stack.pop()
            ordered.append(v)
            n += 1
            for nv in g_rev[v.orig]:
                cnt[nv.orig] -= 1
                if not cnt[nv.orig]:
                    stack.append(nv)

        return ProgGraph(self._proc, ordered)

    def print_graph(self):
        g_rev = defaultdict(lambda: [])
        cnt = {}

        used = self.select_used()
        for v in used:
            cnt[v.orig] = len(v.a)
            for nv in v.a:
                g_rev[nv.orig].append(v)

        stack = []
        for v in used:
            if not cnt[v.orig]:
                stack.append(v)

        nums = {}
        n = 1
        while stack:
            v = stack.pop()
            n += 1
            nums[v.orig] = n
            for nv in g_rev[v.orig]:
                cnt[nv.orig] -= 1
                if not cnt[nv.orig]:
                    stack.append(nv)
            print('{}: {} {}'.format(
                n, v.key, ' '.join(str(nums[nv.orig]) for nv in v.a)))
