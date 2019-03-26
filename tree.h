#ifndef __TREE_H_
#define __TREE_H_

#include <vector>

using namespace std;

/*
template<class T>
class mtree : public vector<T> {

    inline auto &A(int p) {
        return (*this)[p];
    }

    public:
    using vector<T>;

    void zero() {
        fill(this->begin(), this->end(), 0);
    }

    void resize(int n) {
        vector<T>::resize(n);
    }

    auto get(int p) {
        T sum = 0;
        do {
            sum += A(p);
            p -= p&-p;
        } while (p > 0);
        return sum;
    }

    void add(int p, T v) {
        int n = this->size();
        do {
            A(p) += v;
            p += p&-p;
        } while (p <= n);
    }
};
*/

class smtree {
    class node {
        public:

        int s;
        int m;
        
        node(int s, int m)
            : s(s), m(m) {}
    };
    int size;
    vector<node> A;

    public:
    smtree(int _size) {
        size = 1;
        while (size < _size) size *= 2;
        A.resize(size*2+1, node(0, 0));
    }

    // increments interval (b,e>
    // returns max on (b,∞)
    int inc(int b, int e) {
        b += size;
        e += size;

        A[e].s++;
        auto mb = 0;
        auto me = 0;

        int rmx = -1000000000;

        while ((e^b) >> 1) {
            rmx += A[b].s;
            mb += A[b].s;
            A[b].m = mb = max(A[b].m, mb);
            if (!(b&1)) {
                auto &node = A[b+1];
                node.s++;
                node.m++;
                mb = max(node.m, mb);

                rmx = max(rmx, node.m);
            }

            me += A[e].s;
            A[e].m = me = max(A[e].m, me);
            if (  e&1 ) {
                auto &node = A[e-1];
                node.s++;
                node.m++;
                me = max(node.m, me);
            }

            b >>= 1;
            e >>= 1;
        }

        mb = max(mb, me);

        while (b) {
            rmx += A[b].s;
            mb += A[b].s;

            if (!(b&1)) {
                rmx = max(rmx, A[b+1].m);
            }
            A[b].m = mb = max(A[b].m, mb);
            
            b >>= 1;
        }

        return rmx;
    }

    // max of <p,∞)
    int rmx(int p) {
        p += size;

        int m = 0;
        while (p) {
            m += A[p].s;
            if (!(m&1)) {
                m = max(A[p+1].m, m);
            }
            p >>= 1;
        }

        return m;
    }
};

#endif /* __TREE_H_ */

