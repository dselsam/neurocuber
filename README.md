# NeuroCuber

NeuroCuber: training NeuroSAT to make cubing decisions for hard SAT problems.

## Overview

In previous work, we showed that the neural network architecture _NeuroSAT_ can learn to solve SAT problems even when trained only as a classifier to predict satisfiability (see [paper](https://openreview.net/forum?id=HJMC_iA5tm) and [code](https://github.com/dselsam/neurosat) for more details).
We found this result fascinating. However, as we discuss at the end of the paper, it is not clear if this finding will lead to improvements to state-of-the-art SAT solvers.

The main reason for pessimism is that CDCL solvers are already extremely fast and reliable on many classes of important SAT problems. Moreover, many of the problems that are hard for CDCL are so large that they would not even fit in GPU memory. In this work, we do not even try to compete with CDCL solvers on the problems that they excel at, nor do we try to solve the problems that cause CDCL solvers strife due to their size. Instead, we restrict our focus to the (not as common) classes of SAT problems that have been shown to benefit from expensive, hard-engineered branching heuristics, and that are nonetheless moderately sized.

Specifically, we consider the [cube-and-conquer](https://www.cs.utexas.edu/~marijn/publications/cube.pdf) paradigm, whereby a "cuber" puts a lot of effort in choosing the first several variables to branch on, and a CDCL solver "conquers" the subproblems produced by the cuber. This approach is relatively new and is state-of-the-art on many classes of very hard SAT problems. As an example, this approach was recently used to solve the long-open [Boolean Pythagorean Triples Problem](https://www.nature.com/news/two-hundred-terabyte-maths-proof-is-largest-ever-1.19990).

In this work, we investigate whether we can use the NeuroSAT architecture to learn a better cuber.
