#ifndef __OPTIM_H_
#define __OPTIM_H_

#include "tree.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>

#include <boost/python.hpp>

namespace py = boost::python;

template<class Tv>
auto convert_to_vector(py::object &py_list) {
    vector<Tv> r;
    int l = py::len(py_list);
    for (int i=0; i<l; i++) {
        r.push_back(
            py::extract<Tv>(py_list[i])
        );
    }
    return r;
}

typedef double op_time_t;

struct op_descr {
    op_time_t len_t;
    vector<int> ports;
    op_descr(op_time_t len_t, vector<int> &ports)
        : len_t(len_t), ports(ports) {}
};

class proc_state;

class proc_descr {
    friend class proc_state;

    struct mem_level {
        int size;
        int port_n;
        op_time_t load_time;
        mem_level(int size, int port_n, op_time_t load_time)
            : size(size), port_n(port_n), load_time(load_time) {}
    };

    int n_ports;
    vector<op_descr> ops;
    vector<mem_level> mem_descr;

    public:
    proc_descr(int n_ports) : n_ports(n_ports) {}

    auto *get_op(int op_id) {
        return & ops[op_id];
    }

//    auto new_state(int n_ops) {
//        return proc_state(*this, n_ops);
//    }

    int new_mem_level(int size, int port_n, op_time_t load_time) {
        auto r = mem_descr.size();
        mem_descr.emplace_back(size, port_n, load_time);
        return r;
    }

    int new_op(op_time_t len_t, py::object &ports) {
        auto r = ops.size();
        int l = py::len(ports);
        auto v_ports = convert_to_vector<int>(ports);
        ops.emplace_back(len_t, v_ports);
        return r;
    }
};

class proc_state {
    proc_descr & p;
    smtree mt;
    vector<op_time_t> ports_free_time;
    vector<op_time_t> end_t;
    vector<int> last_usage;
    map<int, op_time_t> m_port_map;
    op_time_t op_start_t;

    public:

    proc_state(proc_descr & p, int n_ops) :
        p(p),
        ports_free_time(p.n_ports, 0),
        mt(n_ops+1),
        end_t(n_ops),
        last_usage(n_ops) {
        clear();
    }

    void clear() {
        op_start_t = 0;
        mt.clear();
        fill(ports_free_time.begin(), ports_free_time.end(), 0);
    }

    auto &mem_level_select(int q_n) {
        for (auto &m : p.mem_descr) {
            if (q_n <= m.size)
                return m;
            q_n -= m.size;
        }
        return p.mem_descr.back();
    }

    auto use_mem(int src_v, int op_step) {
        auto src_step = last_usage[src_v];
        int q_n = mt.inc(src_step, op_step);
        auto &m =  mem_level_select(q_n);
        op_start_t = max(op_start_t, end_t[src_v]);
        m_port_map[m.port_n] += m.load_time;
        last_usage[src_v] = op_step;
    }

    auto finish_time() {
        return * max_element(ports_free_time.begin(), ports_free_time.end());
    }

    class op_adder {
        proc_state & p;
        op_descr & op;
        int step_num;

        public:

        op_adder(proc_state &p, op_descr &op, int step_num)
            :p(p), op(op), step_num(step_num) {}

        void use_mem(int src_v) {
            p.use_mem(src_v, step_num);
        }

        void perform(int v) {
            for (const auto &m: p.m_port_map) {
                auto port_n = m.first;
                auto use_t = m.second;
                p.op_start_t = p.ports_free_time[port_n] =
                    max(p.ports_free_time[port_n] + use_t, p.op_start_t);
            }

            int n = -1;
            op_time_t ot = 1e18;
            for (auto pn: op.ports) {
                if (p.ports_free_time[pn] < ot) {
                    ot = p.ports_free_time[pn];
                    n = pn;
                }
            }
            ot = p.op_start_t = max(p.op_start_t, ot);
            p.end_t[v] = p.ports_free_time[n] = ot + op.len_t;
            p.last_usage[v] = step_num;
        }
    };

    auto add_new_op(op_descr &op, int step_num) {
        m_port_map.clear();
        return op_adder(*this, op, step_num);
    }
};

class prog {
    struct single_op_descr {
        op_descr *op;
        int start_pos, end_pos;
        float exp_use;

        inline auto clamp_pos(int pos) {
            if (start_pos > pos) {
                return start_pos
            }
            if (end_pos < pos) {
                return end_pos;
            }
            return pos;
        }
    };

    py::object p_ref;
    proc_descr &p;

    vector<single_op_descr> ops;
    vector<vector<int>> G;
    vector<vector<int>> G_rev;

    vector<int> left;
    proc_state state;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> node_queue;

    public:

    prog(py::object p, py::object nops_list, py::object Gpy)
        : p(py::extract<proc_descr&>(p)), p_ref(p), left(py::len(nops_list)), state(this->p, left.size()) {
        auto n = left.size();
        ops.resize(n);
        G_rev.resize(n);

        for (long i=0; i<n; i++) {
            auto &cur_op = ops[i];
            auto &cur_obj = nops_list[i];
            cur_op.op = (this->p).get_op(
                py::extract<int>(cur_obj.attr("nop"))
            );
            cur_op.start_pos = py::extract<int>(cur_obj.attr("start_pos"));
            cur_op.end_pos = py::extract<int>(cur_obj.attr("end_pos"));
            cur_op.exp_use = py::extract<float>(exp_use.attr("exp_use"));
        }

        for (long i=0; i<n; i++) {
            auto &v = G.emplace_back();
            auto vpy = Gpy[i];
            auto m = py::len(vpy);
            for (long j=0; j<m; j++) {
                v.push_back(py::extract<int>(vpy[j]));
            }
        }

        for (int i=0; i<n; i++) {
            auto &v = G[i];
            for (auto &nv : v) {
                G_rev[nv].push_back(i);
            }
        }
    }

    auto size() {
        return ops.size();
    }

    auto reorder_f(vector<int> &order) {
        int step_num=0;

        state.clear();

        for (int i=0; i<G.size(); i++) {
            int n = left[i] = G[i].size();
            if (!n) {
                node_queue.emplace(ops[i].clamp_pos(order[i]), i);
            }
        }

        while (!node_queue.empty()) {
            auto v = node_queue.top().second; node_queue.pop();
            order[v] = step_num++;
            for (auto vn : G_rev[v]) {
                left[vn] --;
                if (!left[vn]) {
                    node_queue.emplace(ops[vn].clamp_pos(order[vn]), vn);
                }
            }

            auto op_adder = state.add_new_op(* ops[v].op, step_num);
            for (auto vb : G[v]) {
                op_adder.use_mem(vb);
            }
            op_adder.perform(v);
        }

        return state.finish_time();
    }

    void reorder_b(vector<int> &order) {
        int o=G.size();

        //priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> q;

        for (int i=0; i<G_rev.size(); i++) {
            int n = left[i] = G_rev[i].size();
            if (!n) {
                node_queue.emplace(o - ops[i].clamp_pos(order[i]), i);
            }
        }

        while (!node_queue.empty()) {
            auto v = node_queue.top().second; q.pop();
            order[v] = --o;
            for (auto i : G[v]) {
                left[i] --;
                if (!left[i]) {
                    node_queue.emplace(ops[i].clamp_pos(order[i]), i);
                }
            }
        }
    }

//    auto score(vector<int> &order) { ////// !!!!!!!!!!!!!!!!!!
//        vector<int> last_usage(order.size()+1);
//        vector<int> op_step(order.size());
//
//        for (int i=0; i<order.size(); i++)
//            op_step[order[i]] = i+1;
//
//        auto state = p.new_state(order.size());
//
//        for (int step_num=1; step_num<=order.size(); step_num++) {
//            int op_n = order[step_num-1];
//
//            auto op_adder = state.add_new_op(*ops[op_n], step_num);
//            for (auto v : G[op_n]) {
//                auto v_step = op_step[v];
//                op_adder.use_mem(v_step);
//                last_usage[v_step] = step_num;
//            }
//
//            last_usage[step_num] = step_num-1;
//        }
//
//        return state.finish_time();
//    }
};

void test(prog &prg) {
    vector<int> ord;
    float score = 1e18;
    for (int i=0; i<prg.size(); i++) {
        ord.push_back(-i);
    }
    cout << prg.reorder_f(ord) << endl;
    for (int k=0; k<20; k++) {
        for (int j=25; j>=1; j--) {
            for (int i=0; i<ord.size(); i++) {
                auto ord_new = ord;
                ord_new[i]+=3;
                auto score_new = prg.reorder_f(ord_new);
                if (score_new <= score) {
                    score = score_new;
                    ord = ord_new;
                }
            }
        }
        for (int j=25; j>=1; j--) {
            for (int i=ord.size()-1; i>=0; i--) {
                auto ord_new = ord;
                ord_new[i]-=j;
                auto score_new = prg.reorder_f(ord_new);
                if (score_new <= score) {
                    score = score_new;
                    ord = ord_new;
                }
            }
}
    }
    cout << prg.reorder_f(ord) << endl;
    cout << 500 * ord.size();
    cout << endl;
}

#endif /* __OPTIM_H_ */

