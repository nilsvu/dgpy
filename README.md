# dgpy

Python library to solve elliptic partial differential equations with
discontinous Galerkin (DG) schemes. The purpose of this library is to prototype
elliptic DG schemes for the [SpECTRE code](https://github.com/sxs-collaboration/spectre).
`dgpy` provides tools to construct rectilinear domains in 1, 2 or 3 dimensions,
implement elliptic DG schemes on Legendre-Gauss-Lobatto and Legendre-Gauss
grids, solve them with iterative algorithms and visualize the results.

## Installation

1. `git clone` this repository to your machine.
2. `pip install -e /path/to/repository` to install in editable mode, i.e. where
   changes to the repository are reflected in the `pip` installation, or omit
   the `-e` to install in user mode.

## Getting started

Read through the [walkthrough notebook](walkthrough.ipynb) to get started:

- [Walkthrough notebook](walkthrough.ipynb)
