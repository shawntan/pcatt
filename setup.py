from sysconfig import get_path
from setuptools import setup, Extension
from pathlib import Path

PATH_PREFIX = get_path('data')
module1 = Extension(f'greedy_builder',
                    extra_compile_args = ["-O3", "-std=c++20"],
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '10')],
                    include_dirs = [f'{PATH_PREFIX}/include/',
                                    f'{PATH_PREFIX}/include/tbb',
                                    f'{PATH_PREFIX}/include/oneapi'],
                    library_dirs = [f'{PATH_PREFIX}/lib/'],
                    libraries = ['tbb'],
                    sources = ['pcatt/greedy_builder.cpp'])

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="greedtok",
    version="0.13",
    description="Partition Cover Approach to Tokenization",
    author="JP Lim",
    author_email="jiapeng.lim.2021@phdcs.smu.edu.sg",
    license = "MIT",
    setup_requires=['pybind11', 'tbb-devel'],
    url = "https://github.com/PreferredAI/pcatt/",
    download_url = "https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.13.tar.gz",
    ext_modules = [module1],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
