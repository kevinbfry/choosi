from glob import glob
from setuptools import setup, find_packages
from distutils.dir_util import copy_tree
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import sysconfig
import os
import platform
import subprocess


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(" "), stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = open("VERSION", "r").read()

# ENVPATH = os.getenv("CONDA_PREFIX")

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra", "-DNDEBUG", "-O3"]
libraries = []
extra_link_args = []

system_name = platform.system()
if (system_name == "Darwin"):
    omp_prefix = run_cmd("brew --prefix libomp")
    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")
    extra_compile_args += [
        f"-I{omp_include}",
        "-Xclang",
        "-fopenmp",
    ]
    extra_link_args += [f'-L{omp_lib}']
    libraries = ['omp']
    
if (system_name == "Linux"):
    extra_compile_args += ["-fopenmp"]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "choosi.choosi_core",
        sorted(glob("choosi/src/*.cpp")),  # Sort source files for reproducibility
        defines=[],
        include_dirs=[
            "choosi/src",
            "choosi/src/include",
            # os.path.join(ENVPATH, 'include'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        cxx_std=14,
    ),
]

setup(
    name='choosi', 
    version=__version__,
    description='A Python package for SNP-related tools.',
    long_description='',
    author='Kevin Fry',
    author_email='kfry@stanford.edu',
    maintainer='Kevin Fry',
    maintainer_email='kfry@stanford.edu',
    packages=["choosi"], 
    package_data={
        "choosi": [
            "src/**/*.hpp", 
            "src/third_party/**/*",
            "choosi_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)