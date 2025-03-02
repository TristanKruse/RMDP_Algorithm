#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cheapest_insertion.hpp"

namespace py = pybind11;

// Make sure this matches exactly with the name in setup.py
PYBIND11_MODULE(cheapest_insertion, m) {  // This must match the Extension name in setup.py
    m.doc() = "C++ implementation of cheapest insertion for RMDP"; // optional module docstring

    // Bind Location struct
    py::class_<rmdp::Location>(m, "Location")
        .def(py::init<>())
        .def(py::init<double, double>())  // Add constructor for x,y
        .def_readwrite("x", &rmdp::Location::x)
        .def_readwrite("y", &rmdp::Location::y);

    // Bind Stop struct
    py::class_<rmdp::Stop>(m, "Stop")
        .def(py::init<>())
        .def_readwrite("node_id", &rmdp::Stop::node_id)
        .def_readwrite("pickups", &rmdp::Stop::pickups)
        .def_readwrite("deliveries", &rmdp::Stop::deliveries);

    // Bind InsertionResult struct
    py::class_<rmdp::InsertionResult>(m, "InsertionResult")
        .def(py::init<>())
        .def_readwrite("vehicle_id", &rmdp::InsertionResult::vehicle_id)
        .def_readwrite("new_sequence", &rmdp::InsertionResult::new_sequence)
        .def_readwrite("insertion_cost", &rmdp::InsertionResult::insertion_cost);

    // Bind CheapestInsertion class
    py::class_<rmdp::CheapestInsertion>(m, "CheapestInsertion")
        .def(py::init<double, double, double>())
        .def("find_best_insertion", &rmdp::CheapestInsertion::findBestInsertion);
}