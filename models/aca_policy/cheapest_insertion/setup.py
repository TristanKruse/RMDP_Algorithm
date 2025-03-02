from setuptools import setup, Extension
import pybind11
import sys

# Compiler flags
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args.append('/std:c++20')
else:
    extra_compile_args.append('-std=c++20')

ext_modules = [
    Extension(
        "cheapest_insertion",  # This must match the PYBIND11_MODULE name
        sources=["src/bind_cheapest_insertion.cpp", "src/cheapest_insertion.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "src"
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="cheapest_insertion",
    version="0.0.1",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)