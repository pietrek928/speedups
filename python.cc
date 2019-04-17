#include "optim.h"

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(optim) {
    class_<proc_descr, boost::noncopyable>("_proc", init<int>())
        .def("new_mem_level", &proc_descr::new_mem_level)
        .def("new_op", &proc_descr::new_op)
    ;

    class_<prog>("_prog", init<py::object, py::object, py::object>())
    ;

    def("test", test);
}
