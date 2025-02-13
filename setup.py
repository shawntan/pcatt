from sysconfig import get_path
from setuptools import setup, Extension
from pathlib import Path

PATH_PREFIX = get_path("data")
modules = []
for code in ["greedy_builder, greedy_encoder, pco_tokenizer"]:
    modules.append(
        Extension(
            code,
            extra_compile_args=["-O3", "-std=c++23"],
            define_macros=[("MAJOR_VERSION", "0"), ("MINOR_VERSION", "14-beta")],
            include_dirs=[f"{PATH_PREFIX}/include/", f"{PATH_PREFIX}/include/tbb"],
            library_dirs=[f"{PATH_PREFIX}/lib/"],
            libraries=["tbb"],
            sources=[f"pcatt/{code}.cpp"],
        )
    )

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="greedtok",
    version="0.14-beta",
    description="Partition Cover Approach to Tokenization",
    author="JP Lim",
    author_email="jiapeng.lim.2021@phdcs.smu.edu.sg",
    license="MIT",
    setup_requires=["pybind11", "tbb-devel"],
    url="https://github.com/PreferredAI/pcatt/",
    download_url="https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.13.tar.gz",
    ext_modules=modules,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
