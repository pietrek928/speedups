#include "optim.h"

#include <iostream>

#include <map>
#include <queue>

int main() {
    smtree T(32);
    cout << T.inc(1, 3) << endl;
    cout << T.inc(2, 5) << endl;
    cout << T.inc(1, 5) << endl;
    //map<int, float> m;
    //m[1] += 1;
    //cout << m[1] << endl;
    //priority_queue<pair<int, int>> q;
    //q.emplace(1, 2);
    //q.emplace(3, 4);
    //q.pop(); q.pop();
    //cout << q.top().first << endl;
    //cout << q.top() << endl;
    //cout << q.top() << endl;
    return 0;
}

