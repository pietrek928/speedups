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

class proc_descr {

    struct mem_level {
        int size;
        int port_n;
        op_time_t load_time;
        mem_level(int size, int port_n, op_time_t load_time)
            : size(size), port_n(port_n), load_time(load_time) {}
    };

    struct op_descr { // TODO: type checking
        op_time_t len_t;
        vector<int> ports;
        op_descr(op_time_t len_t, vector<int> &ports)
            : len_t(len_t), ports(ports) {}
    };

    int n_ports;
    vector<op_descr> ops;
    vector<mem_level> mem_descr;

    class proc_state {
        proc_descr & p;
        smtree mt;
        vector<op_time_t> ports_free_time; 
        vector<op_time_t> end_t;
        map<int, op_time_t> m_port_map;
        op_time_t op_start_t = 0;

        public:

        proc_state(proc_descr & p, int n_ops) :
            p(p),
            ports_free_time(p.n_ports, 0),
            mt(n_ops+1),
            end_t(n_ops) {
        }

        auto &mem_level_select(int q_n) {
            for (auto &m : p.mem_descr) {
                if (q_n <= m.size)
                    return m;
                q_n -= m.size;
            }
            return p.mem_descr.back();
        }

        auto use_mem(int src_step, int op_step) {
            int q_n = mt.inc(src_step, op_step);
            auto &m =  mem_level_select(q_n);
            op_start_t = max(op_start_t, end_t[src_step]);
            m_port_map[m.port_n] += m.load_time;
        }

        auto finish_time() {
            return max_element(ports_free_time.begin(), ports_free_time.end());
        }

        class op_adder {
            proc_state & p;
            op_descr & op;
            int step_num;
            public:

            op_adder(proc_state &p, op_descr &op, int step_num)
                :p(p), op(op), step_num(step_num) {}

            void use_mem(int src_step) {
                p.use_mem(src_step, step_num);
            }

            void perform() {
                for (auto &m: p.m_port_map) {
                    auto port_n = m.first;
                    auto use_t = m.second;
                    p.op_start_t = p.ports_free_time[port_n] =
                        max(p.ports_free_time[port_n]+use_t, p.op_start_t);
                }

                int n = -1;
                op_time_t ot = 1e18;
                for (auto pn: op.ports) {
                    if (p.ports_free_time[pn] < ot) {
                        ot = p.ports_free_time[pn];
                        n = pn;
                    }
                }
                p.op_start_t = max(p.op_start_t, ot);
                p.end_t[step_num] = p.ports_free_time[n] = ot + op.len_t;
            }
        };

        auto add_new_op(int op_n, int step_num) {
            m_port_map.clear();
            return op_adder(*this, p.ops[op_n], step_num);
        }
    };

    public:
    proc_descr(int n_ports) : n_ports(n_ports) {}

    auto new_state(int n_ops) {
        return proc_state(*this, n_ops);
    }

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

class prog {
    py::object p_ref;
    proc_descr &p;

    vector<vector<int>> G;
    vector<vector<int>> G_rev;

    public:

    prog(py::object p, py::object Gpy) : p(py::extract<proc_descr&>(p)), p_ref(p) {
        py::size_t n = py::len(Gpy);
        for (py::size_t i=0; i<n; i++) {
            auto &v = G.emplace_back();
            auto &vpy = Gpy[i];
            py::size_t m = py::len(vpy);
            for (py::size_t j=0; j<m; j++) {
                v.push_back(extract<int>(vpy));
            }
        }

        G_rev.resize(n);
        for (int i=0; i<n; i++) {
            auto &v = G_rev[i];
            for (auto &nv : v) {
                G_rev[nv].push_back(v);
            }
        }
    }

    void reorder_f(vector<int> &order) {
        int o=0;

        vector<int> left(order.size());
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;

        for (int i=0; i<G.size(); i++) {
            int n = left[i] = G[i].size();
            if (!n) {
                q.emplace(order[i], i);
                order[i] = o++;
            }
        }

        while (!q.empty()) {
            auto v = q.top().second; q.pop();
            for (auto i : G_rev[v]) {
                left[i] --;
                if (!left[i]) {
                    q.emplace(order[i], i);
                    order[i] = o++;
                }
            }
        }
    }

    void reorder_b(vector<int> &order) {
        int o=G.size();

        vector<int> left(order.size());
        priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> q;
        
        for (int i=0; i<G_rev.size(); i++) {
            int n = left[i] = G_rev[i].size();
            if (!n) {
                q.emplace(order[i], i);
                order[i] = --o;
            }
        }

        while (!q.empty()) {
            auto v = q.top().second; q.pop();
            for (auto i : G[v]) {
                left[i] --;
                if (!left[i]) {
                    q.emplace(order[i], i);
                    order[i] = --o;
                }
            }
        }
    }

    auto score(vector<int> &order) {
        int pos = 0;
        vector<int> last_usage(order.size()+1);
        vector<int> op_step(order.size());

        for (int i=0; i<order.size(); i++)
            op_step[order[i]] = i+1;

        auto state = p.new_state(order.size());

        for (int step_num=1; step_num<=order.size(); step_num++) {
            int op_n = order[step_num-1];

            auto op_adder = state.add_new_op(op_n, step_num);
            for (auto v : G[op_n]) {
                auto v_step = op_step[v];
                op_adder.use_mem(v_step);
                last_usage[v_step] = step_num;
            }

            last_usage[step_num] = step_num-1;
        }

        return state.finish_time();
    }
};

#endif /* __OPTIM_H_ */

