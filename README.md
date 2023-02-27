# Levenberg–Marquardt and line-search iterated posterior linearisation smoothing

This is the code base for the pre-print [Posterior linearisation smoothing with robust iterations](https://arxiv.org/abs/2112.03969).
It also provides implementations for related papers

## Setup

The python dependencies are managed by [`pipenv`](https://pipenv.pypa.io/en/latest/),
see `Pipfile` for the requirements.

```
# Get the source code
git clone git@github.com:jackonelli/post_lin_smooth.git
cd post_lin_smooth
# Start a shell with a python virtual env. with the needed deps.
pipenv shell
```

## Recreate experiments:

Results published in the the paper can be reproduced by running the corresponding script in the `exp` sub-dir.
Using a fixed seed will guarantee exactly replicated data and conditions.

- Figure 2. Simulated coordinated turn with bearings only measurements
  ```bash
  python exp/coord_turn/realisation.py --meas_type bearings --num_iter 10
  ```
- Figure 4. Visualisation of a single realisation of the CT experiment with varying bearings only measurements.
  ```bash
  python exp/coord_turn/realisation.py --meas_type bearings --var_sensors --num_iter 10
  ```

For previous results that this code reproduces a fixed seed is not possible (the previous papers' experiments are exclusively implemented in matlab).
For them the actual data used in those trials are included in this repo (the `data` sub-dir).

## Test coverage

Explicit unit testing as such, is limited at the moment.
There is, however, multiple larger tests, akin to integration tests.
These are automated verification that the library implementation faithfully reproduces the results published in the papers describing the various implemented filter/smoother.
All filters and smoothers (more or less) use the same base methods in the end.
A test veryfying that a smoother in this library produces the same results as those in peer reviewed papers, is a strong indication that the underlying base methods are correct.
Especially when every value is accurate over a long sequence, in multiple dimensions and for many iterations.

To run all the tests, run (from the repo. root):

```bash
python -m unittest
```
