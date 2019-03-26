#include "optim.h"

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(optim) {
    class_<proc_descr, boost::noncopyable>("_proc", init<int>())
        .def("new_mem_level", &proc_descr::new_mem_level)
        .def("new_op", &proc_descr::new_op)
    ;
}

BOOST_PYTHON_MODULE(prog) {
    class_<proc_descr, boost::noncopyable>("_prog", init<py::object p, py::object Gpy>())
    ;
}
