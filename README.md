# NeuroCuber

NeuroCuber: training NeuroSAT to make cubing decisions for hard SAT problems.

## Overview

In previous work, we showed that the neural network architecture _NeuroSAT_ can learn to solve SAT problems even when trained only as a classifier to predict satisfiability (see [paper](https://openreview.net/forum?id=HJMC_iA5tm) and [code](https://github.com/dselsam/neurosat) for more details).
We found this result fascinating. However, as we discuss at the end of the paper, it is not clear if this finding will lead to improvements to state-of-the-art SAT solvers.

The main reason for pessimism is that CDCL solvers are already extremely fast and reliable on many classes of important SAT problems. Moreover, many of the problems that are hard for CDCL are so large that they would not even fit in GPU memory. In this work, we do not even try to compete with CDCL solvers on the problems that they excel at, nor do we try to solve the problems that cause CDCL solvers strife due to their size. Instead, we restrict our focus to the (not as common) classes of SAT problems that have been shown to benefit from expensive, hard-engineered branching heuristics, and that are nonetheless moderately sized.

Specifically, we consider the [cube-and-conquer](https://www.cs.utexas.edu/~marijn/publications/cube.pdf) paradigm, whereby a "cuber" puts a lot of effort in choosing the first several variables to branch on, and a CDCL solver "conquers" the subproblems produced by the cuber. This approach is relatively new and is state-of-the-art on many classes of very hard SAT problems. As an example, this approach was recently used to solve the long-open [Boolean Pythagorean Triples Problem](https://www.nature.com/news/two-hundred-terabyte-maths-proof-is-largest-ever-1.19990).

In this work, we investigate whether we can use the NeuroSAT architecture to learn a better cuber.

## Evaluating cubers

Ultimately, we will only consider NeuroCuber a success if we can use it to solve real problems of interest more efficiently than would otherwise be possible. However, we want to be able to estimate NeuroSAT's ability efficiently without needing to solve entire SAT problems, especially during early stages of the project.

In 1975, Knuth introduced [a simple, unbiased estimator for the size of a search tree](https://pdfs.semanticscholar.org/94ce/5bdf77af8693df0d525010850ab6faf7e290.pdf). For binary trees such as ours, this method reduces to averaging the exponentiated lengths of random paths in the tree. Yet as Knuth discusses, this method may have extremely high variance. The solution he suggests and that we adopt in this work is to use importance sampling. Specifically, in addition to training NeuroSAT to make cubing decisions, we train it to predict which branch is the harder one, and use these predictions for importance sampling.
