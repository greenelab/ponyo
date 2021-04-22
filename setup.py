import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

test_pckgs = [
    "pytest",
    "nbval",
    "umap-learn",
    "plotnine",
    "coverage<5.0",
    "pytest-cov",
    "coveralls==2.2.0",
]

extras = {
    "test": test_pckgs,
}

setup(
    name="ponyo",
    version="0.4",
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
        "python>=3.5, <3.8"
        "pandas",
        "numpy",
        "keras==2.3.1",
        "tensorflow==1.15.4",
        "scikit-learn",
        "h5py<3",
    ],
    tests_require=test_pckgs,
    extras_require=extras,
)
