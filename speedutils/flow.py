from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, List, Optional, Set

from . import optim
from .graph import GraphOptim
from .graphval import GraphVal, OpScope
from .vtypes import OpDescr, VType


class ScopeDescr:
    def __init__(self, exp_use: float):
        self.exp_use: float = exp_use
        self.nodes: List['GraphVal'] = []

    def append(self, v: 'GraphVal'):
        self.nodes.append(v)


class NodeMapper:
    def __init__(self, alias_mapper):
        self._alias_mapper = alias_mapper
        self._d = {}
        self._nd = defaultdict(int)

    def set(self, v: GraphVal, n: str = None):
        if not n:
            prefix = v.name_prefix or 'v'
            n = f'{prefix}{self._nd[prefix]}'
            self._nd[prefix] += 1
        self._d[v.orig] = n

    def get(self, v: GraphVal):
        try:
            return self._d[self._alias_mapper(v).orig]
        except KeyError:
            raise RuntimeError(f'No name registered for {v}, {v.__dict__}')


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
        self._op_idx: Dict[str, GraphVal] = {}
        self._proc: optim._proc = p
        self._use_stack = [1.0]
        self._scope_list: List[ScopeDescr] = []
        self._orig_aliases = {}

        self.new_scope()

    def find_op(self, n: str, a: Iterable[GraphVal]) -> OpDescr:
        t = tuple(v.type_name for v in a)
        op_n = f'{n}Y{"X".join(t)}'
        try:
            return self.ops[op_n]
        except KeyError:
            reorder_op_n = f'{n}Y{"X".join(sorted(t))}'
            try:
                op = self.ops[reorder_op_n]
                if not op.ordered:
                    return op
            except KeyError:
                raise ValueError(
                    f'No operation `{op_n}` on types'
                    f'{tuple(type(v) for v in a)}'
                )

    def find_const_op(self, t: VType):
        op_n = f'constY{t.type_name}'
        return self.ops[op_n]

    def find_load_op(self, t: VType):
        op_n = f'loadY{t.type_name}'
        return self.ops[op_n]

    def find_cvt_op(self, v: GraphVal, t: VType):
        op_n = f'cvtY{v.type_name}X{t.type_name}'
        return self.ops[op_n]

    def find_store_op(self, t: VType):
        op_n = f'storY{t.type_name}'
        return self.ops[op_n]

    def find_spec_op(self, name, t: VType):
        op_n = f'{name}Y{t.type_name}'
        try:
            return self.ops[op_n]
        except KeyError:
            raise ValueError(f'No typespec-operation `{op_n}` on `{name}` for type {t}')

    def add_node(self, v: GraphVal) -> GraphVal:
        k = v.key
        # TODO: check and add parent nodes
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

    def get_scope_n(self, *a: GraphVal) -> int:
        scopes = tuple(
            v.scope_n for v in a
        )
        try:
            return max(scopes)
        except ValueError:
            return len(self._scope_list) - 1

    # def op(self, n, *a: GraphVal) -> GraphVal:
    #     return self.add_node(
    #         OpNode(self, n, a)
    #     )

    # def cvt(self, v: GraphVal, t: VType) -> GraphVal:
    #     return self.add_node(
    #         CvtNode(self, v, t)
    #     )
    #
    # def load(self, t: VType, arr: GraphVal, idx: GraphVal) -> GraphVal:
    #     return self.add_node(
    #         LoadNode(self, t, arr, idx)
    #     )

    # def store(self, v: GraphNode, arr: GraphNode, idx) -> GraphNode:
    #     return self.add_node(
    #         StoreNode(self, v, arr, idx)
    #     )

    # def const(self, t: VType, v) -> GraphVal:
    #     # v = self._format_const(t, v)
    #     return self.add_node(
    #         ConstNode(self, t, v)
    #     )

    # def var(self, t: VType, name: str, start_scope=True) -> GraphNode:
    #     v = VarNode(self, t, name)
    #     if start_scope:
    #         v.scope_n = 0
    #     return self.add_node(v)

    # def zero(self, t: VType) -> GraphVal:
    #     ret = OpNode(self, 'zeroY{}'.format(t), ())
    #     ret.num_attrs.add('zero')
    #     self.add_node(ret)
    #     return ret
    #
    # def one(self, t: VType) -> GraphVal:
    #     ret = self.const(t, 1.0)
    #     ret.num_attrs.add('one')
    #     self.add_node(ret)
    #     return ret
    #
    # def bind_scope(self, v: GraphVal) -> GraphVal:
    #     vscoped = v.reset_orig()
    #     vscoped.scope_n = self.get_scope_n()
    #     return vscoped

    # def sep(self, v: GraphNode) -> GraphNode:
    #     return self._add_n(
    #         SepNode(self, v)
    #     )

    def get_alias(self, v: GraphVal) -> GraphVal:
        return self._orig_aliases.get(v.orig) or v

    def add_alias(self, oldv: GraphVal, newv: GraphVal):
        self._orig_aliases[newv.orig] = self.get_alias(oldv)

    def node_mapper(self) -> NodeMapper:
        return NodeMapper(self.get_alias)

    def select_used(self) -> Set[int]:
        stack = []
        used = set()

        for v in self._op_idx.values():
            v = self.get_alias(v)
            if not v.has_output:
                stack.append(v)

        while stack:
            v = stack.pop()
            used.add(v.orig)

            for nv in v.a:
                nv = self.get_alias(nv)
                if nv.orig not in used:
                    stack.append(nv)

        return used

    def optim_graph(self) -> GraphOptim:
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
        nums = self.node_mapper()
        for v in self.used_ordered():
            nums.set(v)
            args = ' '.join(map(nums.get, v.a))
            print(f'{nums.get(v)}: {v.key} {args}')

    def gen_code(self, ord: Optional[Iterable[int]] = None):
        nodes = self.used_ordered()

        if ord is None:
            ord = range(len(nodes))

        nums = self.node_mapper()
        for i in ord:
            v = nodes[i]
            nums.set(v, v.var_name)
            v.print_op(nums)
