# Bayesian filtering and smoothing

This is the code base for the paper "Robust posterior linearisation smoothing".
It also provides implementations for related papers

## Setup

The python dependencies are managed by [`pipenv`](https://pipenv.pypa.io/en/latest/)

```
# Get the source code
git clone git@github.com:jackonelli/post_lin_smooth.git
cd post_lin_smooth
# Start a shell with a python virtual env. with the needed deps.
pipenv shell
```

### Data

Results first achieved for the paper "Paper link" can be reproduced by running the corresponding script in the `exp` sub-dir.
Using a fixed seed will guarantee exactly replicated data and conditions.

For previous results that this code reproduces a fixed seed is not possible (the previous papers' experiments are exclusively implemented in matlab). For them the actual data used in those trials are included in this repo (the `data` sub-dir).

## Test coverage

Explicit unit testing as such, is limited at the moment.
There is, however, a few larger tests, akin to integration tests.
These are automated verification that the library implementation faithfully reproduces the results published in the papers describing the implemented filter/smoother.
All filters and smoothers (more or less) use the same base methods in the end.
A test veryfying that a smoother in this library produces the same results as those in a peer reviewed paper, is a strong indication that the underlying base methods are correct.
Especially when every value is accurate over a long sequence, in multiple dimensions for many iterations.
