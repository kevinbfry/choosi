from glob import glob
from setuptools import setup, find_packages
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

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += [
    "-g0",
    "-Wall", 
    "-Wextra", 
    "-DNDEBUG", 
    "-O3",
]
include_dirs = [
    "choosi/src",
    "choosi/src/include",
]
libraries = []
library_dirs = []
runtime_library_dirs = []

# check if conda environment activated
if "CONDA_PREFIX" in os.environ:
    conda_prefix = os.environ["CONDA_PREFIX"]
# check if micromamba environment activated (CI)
elif "MAMBA_ROOT_PREFIX" in os.environ:
    conda_prefix = os.path.join(os.environ["MAMBA_ROOT_PREFIX"], "envs/choosi")
else:
    conda_prefix = None

# add include, include/eigen3, adelie/src/include
if not (conda_prefix is None):
    conda_include_path = os.path.join(conda_prefix, "include")
    eigen_include_path = os.path.join(conda_include_path, "eigen3")
    adelie_path = run_cmd(f"find {conda_prefix} -name adelie").split()[0]
    adelie_include_path = os.path.join(adelie_path, "src/include")
    include_dirs += [
        conda_include_path,
        eigen_include_path,
        adelie_include_path,
    ]

system_name = platform.system()
if system_name == "Darwin":
    # if user provides OpenMP install prefix (containing lib/ and include/)
    if "OPENMP_PREFIX" in os.environ and os.environ["OPENMP_PREFIX"] != "":
        omp_prefix = os.environ["OPENMP_PREFIX"]

    # else if conda environment is activated
    elif not (conda_prefix is None):
        omp_prefix = conda_prefix
    
    # otherwise check brew installation
    else:
        # check if OpenMP is installed
        no_omp_msg = (
            "OpenMP is not detected. "
            "MacOS users should install Homebrew and run 'brew install libomp' "
            "to install OpenMP. "
        )
        try:
            libomp_info = run_cmd("brew info libomp")
        except:
            raise RuntimeError(no_omp_msg)
        if "Not installed" in libomp_info:
            raise RuntimeError(no_omp_msg)

        # grab include and lib directory
        omp_prefix = run_cmd("brew --prefix libomp")

    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")

    # augment arguments
    include_dirs += [omp_include]
    extra_compile_args += [
        "-Xpreprocessor",
        "-fopenmp",
    ]
    runtime_library_dirs += [omp_lib]
    library_dirs += [omp_lib]
    libraries += ['omp']
    
if system_name == "Linux":
    extra_compile_args += [
        "-fopenmp", 
        "-march=native",
   ]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "choosi.choosi_core",
        sorted(glob("choosi/src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
    ),
]

setup(
    name='choosi', 
    version=__version__,
    description='',
    long_description='',
    author='Kevin Fry',
    author_email='kfry@stanford.edu',
    maintainer='Kevin Fry',
    maintainer_email='kfry@stanford.edu',
    packages=["choosi"], 
    package_data={
        "choosi": [
            "src/**/*.hpp",
            "src/**/*.cpp",
            "choosi_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)