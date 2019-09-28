from itertools import chain
from typing import Dict, Iterable, List, Set, Optional, Tuple, NamedTuple

import optim

from gnode import OpNode, ConstNode, StoreNode, LoadNode, CvtNode, GNode, OpDescr, VarNode, OpScope, CodeNode
from graph import GraphOptim
from utils import str_list, format_nodes, flush_attrs
from vtypes import VType


class ScopeDescr:
    def __init__(self, exp_use: float):
        self.exp_use = exp_use
        self.nodes = []

    def append(self, v: 'GNode'):
        self.nodes.append(v)


class FlowGraph:
    def __init__(self, mem_levels, ops):
        self.ops: Dict[str, OpDescr] = {}
        ports = set(sum(
            (o[3] for o in ops),
            tuple(m[2] for m in mem_levels)
        ))
        p = optim._proc(len(ports))
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
        self._proc: optim._proc = p
        self._use_stack = [1.0]
        self._scope_list = []

        self.new_scope()

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
            self._scope_list[v.scope_n].append(v)
            return v

    def start_use_block(self, exp_use):
        self._use_stack.append(self._use_stack[-1] * exp_use)
        self.new_scope()

    def end_use_block(self):
        self._use_stack.pop()
        self.new_scope()

    def new_scope(self):
        self._scope_list.append(
            ScopeDescr(
                exp_use=self._use_stack[-1]
            )
        )

    def get_scope_n(self, *a: GNode) -> int:
        scopes = tuple(
            v.scope_n for v in a
        )
        try:
            return max(scopes)
        except ValueError:
            return len(self._scope_list) - 1

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

    def var(self, t: VType, name: str):
        return self._add_n(
            VarNode(self, t, name)
        )

    def zero(self, t: VType):
        return self._add_n(
            OpNode(self, 'zeroY{}'.format(t), ())
        )

    def stationary_code(self, code: str, *a: GNode):
        if self._scope_list[-1].nodes:
            self.new_scope()
        return self._add_n(
            CodeNode(self, code, a)
        )

    def select_used(self) -> Set[int]:
        stack = {}
        used = set()

        for v in self._op_idx.values():
            if v.type is None:
                stack[v.orig] = v

        while stack:
            o, v = stack.popitem()
            used.add(o)

            for nv in v.a:
                if nv.orig not in used:
                    stack[nv.orig] = nv

        return used

    def optim_graph(self):
        ordered = []
        op_scopes = []
        used = self.select_used()
        for scope in self._scope_list:
            scope_used = [
                v for v in scope.nodes
                if v.orig in used
            ]
            op_scopes.append(
                OpScope(
                    start_pos=len(ordered),
                    end_pos=len(ordered) + len(scope_used),
                    exp_use=scope.exp_use
                )
            )
            ordered.extend(scope_used)

        return GraphOptim(self._proc, ordered, op_scopes)

    def used_ordered(self):
        used = self.select_used()
        return tuple(
            v for v in
            chain(*(
                scope.nodes for scope in self._scope_list
            ))
            if v.orig in used
        )

    def print_graph(self):
        nums = {}
        n = 0
        for v in self.used_ordered():
            n += 1
            nums[v.orig] = n
            print('{}: {} {}'.format(
                n, v.key, ' '.join(format_nodes(nums, v.a))
            ))

    def gen_code(self, ord: Optional[Iterable[int]] = None):
        nodes = self.used_ordered()

        if ord is None:
            ord = range(len(nodes))

        nums = {}
        for i in ord:
            v = nodes[i]
            nums[v.orig] = i
            v.print_op(nums)
            # !!!!!!!!
