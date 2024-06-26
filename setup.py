import os
from setuptools import setup, find_namespace_packages
from setuptools import find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="codeExplainer",
    version="0.0.1",
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    description=("A part of the codeExplainer package"),
    license="MIT",
    keywords="example documentation tutorial",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT Approved :: MIT License",
    ],
)
