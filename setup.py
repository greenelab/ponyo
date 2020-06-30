import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="ponyo",
    version="0.1",
    description="Install functions to simulate gene expression compendia",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/greenelab/ponyo",
    author="Alexandra Lee",
    author_email="alexjlee.21@gmail.com",
    license="BSD 3-Clause",
    packages=["ponyo"],
    zip_safe=False,
    install_requires=[
        "pandas",
        "numpy",
        "random",
        "glob",
        "keras",
        "tensorflow",
        "sklearn",
        "joblib",
        "rpy2",
        "r-base>=3.6.0",
        "bioconductor-limma" "bioconductor-sva",
    ],
)
