# Black-box cubature

Run `./run.sh` from the submission directory. It uses the Python, NumPy, and
SciPy already installed in the evaluation image. No compilation or network
access is needed.

The program reads an `init` message, requests point values in batches of at
most 512, and writes a single `result`. Its point counter charges each batch
before the request is sent. The seed only determines the sampling designs.

The solver starts with a scrambled Sobol net. It tests structural models
using values from that net and subsequent probes. Separable functions use
adaptive one-dimensional quadrature. For periodic or projected functions the
solver uses lattice rules or projection moments. For piecewise-polynomial
ridges it estimates interfaces from restrictions to lines and integrates
validated truncated-power models. When a structural model is insufficient,
the solver estimates residuals using regression-assisted Sobol sampling,
embedded cubature rules, or defensive importance sampling.
