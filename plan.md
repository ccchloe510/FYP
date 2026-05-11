# Joint Dictionary Learning With Auxiliary-Variable Hinge Coupling

## Purpose

This project studies a joint optimization model that combines:

- sparse dictionary-based representation learning, and
- linear large-margin classification

within a single objective function.

The implementation is fixed to the following experimental setting:

- dataset: `MNIST`
- binary task: `3 vs 8`
- input normalization: values scaled to `[0,1]`
- dictionary constraint: `D in [0,1]^{d x m}`
- sparsity imposed only on `C`
- optimization method: proximal gradient with backtracking

The central research goal is to evaluate whether learning the representation and
the classifier jointly yields a better discriminative representation than a
two-stage pipeline in which dictionary learning and classification are trained
separately.

## Why The Problem Is Joint

The model is joint because the representation variables `(D,C)` and the
classifier variables `(w,b)` are optimized simultaneously inside one coupled
objective.

The reconstruction term

$$
\frac{1}{2}\|X - DC\|_F^2
$$

forces the dictionary `D` and code matrix `C` to explain the data.

The classifier depends directly on each code vector `c_j` through the margin
residual

$$
r_j(C,w,b) = 1 - y_j(w^\top c_j + b).
$$

Therefore the codes are not learned only for reconstruction. They are also
shaped by the classification objective. Likewise, updates to `(w,b)` depend on
the current representation `C`, so the classifier is learned from the evolving
dictionary representation rather than from fixed precomputed features.

This differs from a separate pipeline:

- separate dictionary learning + SVM:
  first learn `(D,C)`, then fix `C` and train `(w,b)`
- joint method:
  optimize `(D,C,w,b)` together through one coupled objective

## Data And Variables

Let

$$
X = [x_1,\dots,x_n] \in [0,1]^{d \times n},
\qquad
y \in \{-1,+1\}^n.
$$

The optimization variables are

$$
D \in [0,1]^{d \times m},
\qquad
C \in \mathbb{R}^{m \times n},
\qquad
w \in \mathbb{R}^m,
\qquad
b \in \mathbb{R},
\qquad
u \in \mathbb{R}^n.
$$

Here:

- `D` is the dictionary
- `C = [c_1,\dots,c_n]` is the code matrix
- `w,b` define a linear classifier in code space
- `u` is an auxiliary variable introduced to decouple the hinge penalty from
  the classifier residual

## Original Coupled Objective

Before introducing the auxiliary variable, the intended coupled objective is

$$
\min_{C,D,w,b}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,r_j(C,w,b))
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

The terms have the following roles:

- `1/2 ||X - DC||_F^2` is the reconstruction loss for dictionary learning
- `gamma/2 ||w||_2^2` is the classifier `L2` regularization
- `eta/2 sum_j max(0,r_j)` is the hinge-style classification penalty
- `mu ||C||_1` promotes sparse codes
- `delta_[0,1](D)` enforces the box constraint on the dictionary

## Auxiliary-Variable Reformulation

To separate the hinge term from the classifier residual during optimization, we
introduce an auxiliary variable `u` with one scalar `u_j` per sample, and define

$$
q_j(C,w,b,u) = u_j - r_j(C,w,b).
$$

The implemented optimization problem is

$$
\min_{C,D,w,b,u}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_{j=1}^n q_j(C,w,b,u)^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

This formulation should be interpreted carefully.

### Interpretation Of `u`

`u` is not a nonnegative slack variable.

Instead, `u` is an auxiliary copy of the margin residual `r_j(C,w,b)`. Since
the margin residual may be either positive or negative, `u_j` must also be
allowed to take either sign. For this reason:

- `u in R^n`, not `R_+^n`
- there is no indicator term `delta_{R_+^n}(u)`
- `u` should not be described as a slack variable in the thesis

The quadratic penalty

$$
\frac{\rho}{2}\sum_j (u_j-r_j)^2
$$

forces `u_j` to remain close to `r_j(C,w,b)`, while

$$
\frac{\eta}{2}\sum_j \max(0,u_j)
$$

applies the hinge penalty to the auxiliary variable.

### Relation To The Original Hinge Objective

This auxiliary-variable formulation is a penalty-based split model.

It is not the same as imposing the exact hard constraint `u_j = r_j(C,w,b)`.
Instead, equality is encouraged by the quadratic penalty with weight `rho`.
Consequently:

- for large `rho`, the auxiliary variable is forced more tightly toward the
  true margin residual
- for finite `rho`, the model should be understood as an auxiliary-variable
  penalty formulation that approximates the original hinge coupling

In thesis writing, the mathematically accurate wording is:

> `u` is an auxiliary variable introduced to decouple the hinge term from the
> classifier residual. The quadratic penalty enforces `u_j ≈ r_j(C,w,b)`, and
> the hinge penalty `max(0,u_j)` therefore approximates the hinge penalty on the
> margin residual.

The wording to avoid is:

> `u` is a nonnegative slack variable.

## Composite Proximal-Gradient Form

Let

$$
x = (C,D,w,b,u).
$$

The objective is written in composite form

$$
F(x) = f(x) + h(x),
$$

where the smooth part is

$$
f(C,D,w,b,u)
= \frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_{j=1}^n q_j(C,w,b,u)^2,
$$

and the nonsmooth part is

$$
h(C,D,u)
= \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

There is no nonnegativity constraint on `u`.

## Gradients Of The Smooth Part

Define

$$
E = DC - X,
\qquad
q = [q_1,\dots,q_n]^\top,
\qquad
s = q \odot y.
$$

Then the gradients of the smooth part are

$$
\nabla_C f = D^\top E + \rho w s^\top,
$$

$$
\nabla_D f = E C^\top,
$$

$$
\nabla_w f = \gamma w + \rho C s,
$$

$$
\nabla_b f = \rho \mathbf{1}^\top s,
$$

$$
\nabla_u f = \rho q.
$$

These are the derivatives implemented in the codebase.

## Proximal Operators

### Sparse-Code Prox

For the `L1` term on `C`,

$$
\operatorname{prox}_{\alpha \mu \|\cdot\|_1}(\widetilde{C})
= \operatorname{sign}(\widetilde{C})
\odot \max(|\widetilde{C}|-\alpha\mu,0).
$$

### Dictionary Prox

For the box constraint on `D`,

$$
\operatorname{prox}(\widetilde{D})
= \operatorname{clip}(\widetilde{D},0,1).
$$

### Auxiliary-Variable Prox

For the nonsmooth scalar function

$$
\tau \max(0,u),
\qquad
\tau = \alpha \frac{\eta}{2},
$$

the elementwise proximal operator is

$$
\operatorname{prox}(\widetilde{u}) =
\begin{cases}
\widetilde{u}, & \widetilde{u} < 0, \\
0, & 0 \le \widetilde{u} \le \tau, \\
\widetilde{u} - \tau, & \widetilde{u} > \tau.
\end{cases}
$$

This is important: negative values are preserved, because `u` is allowed to be
negative.

## Backtracking Condition

The solver uses proximal gradient with backtracking for the composite objective
`F = f + h`.

The acceptance test must be based on the quadratic upper model of the smooth
part:

$$
F(x^{+})
\le
f(x)
+ \langle \nabla f(x), x^{+}-x \rangle
+ \frac{1}{2\alpha}\|x^{+}-x\|^2
+ h(x^{+}).
$$

Equivalently, after the proximal step has handled `h`, it is enough to verify

$$
f(x^{+})
\le
f(x)
+ \langle \nabla f(x), x^{+}-x \rangle
+ \frac{1}{2\alpha}\|x^{+}-x\|^2.
$$

This is the correct backtracking condition for proximal gradient on a composite
objective. A plain total-objective sufficient-decrease test is not the intended
rule here.

## Initialization Rules

The current implementation uses the following initialization:

- `D`:
  sample `m` training columns and clip them into `[0,1]`
- `C`:
  initialize to zero
- `w,b`:
  initialize to zero
- `u`:
  initialize as the current margin residual

That is,

$$
u^{(0)} = r(C^{(0)}, w^{(0)}, b^{(0)}).
$$

This is consistent with the interpretation of `u` as an auxiliary copy of the
margin residual.

## Code Contracts

- `load_mnist_binary(...) -> X_train, y_train, X_val, y_val, X_test, y_test`
  - `X_*` has shape `(d, n_split)`
  - `y_*` has shape `(n_split,)`
- `initialize_params(X_train, y_train, m, seed) -> params`
  - returns `D, C, w, b, u`
- `objective(params, X, y, hyper) -> dict`
  - returns total objective and component values
- `gradients(params, X, y, hyper) -> dict`
  - returns gradients for `C, D, w, b, u`
- `fit_joint_pg(...) -> result`
  - returns final parameters, optimization history, and termination status
- `fit_raw_svm(...) -> result`
- `fit_separate_dict_svm(...) -> result`

## Baseline For Comparison

The main comparison baseline is a separate two-stage method:

### Stage 1

$$
\min_{D,C}
\frac{1}{2}\|X - DC\|_F^2
+ \mu\|C\|_1
+ \delta_{[0,1]}(D)
$$

### Stage 2

Train a linear SVM on the learned codes.

This baseline is used to evaluate whether the joint coupling of representation
and classifier improves performance relative to learning the dictionary first
and classification second.

## Experimental Outputs

The joint solver should provide:

- final `D, C, w, b, u`
- objective history
- component history for reconstruction, classifier regularization, quadratic
  penalty, hinge term, and sparsity term
- step-size history
- convergence status

The main evaluation metrics across methods are:

- validation accuracy
- test accuracy
- reconstruction error
- code sparsity
- runtime
- objective trajectory versus iteration

## Recommended Thesis Language

The following wording is safe and mathematically aligned with the implemented
model:

> We consider a joint dictionary-learning and classification objective in which
> an auxiliary variable is introduced to decouple the hinge penalty from the
> classifier residual. A quadratic penalty term enforces closeness between the
> auxiliary variable and the margin residual, while sparsity is imposed on the
> code matrix and box constraints are imposed on the dictionary.

The following wording should be avoided:

- "`u` is a nonnegative slack variable"
- "the auxiliary formulation is exactly identical to the original hinge model"

The more accurate statement is:

> the auxiliary formulation is a penalty-based split model that recovers the
> original coupling more closely as the quadratic penalty parameter increases.
