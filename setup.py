#!/usr/bin/env python

from distutils.core import setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='dgpy',
    version='0.1',
    description="Infrastructure to solve elliptic partial differential equations with discontinous Galerkin schemes",
    author="Nils L. Fischer",
    author_email="hello@nilsleiffischer.de",
    url="https://github.com/nilsleiffischer/dgpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['dgpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
