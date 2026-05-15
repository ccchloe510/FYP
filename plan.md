# Joint Dictionary Learning With Auxiliary-Variable Hinge Coupling

## 1. Project Overview

This project studies a joint optimization formulation for dictionary-based
feature learning and SVM-like binary classification.

The main research question is:

> Can a representation and a classifier be learned jointly so that the learned
> features are representative of the input, discriminative for classification,
> and explainable through dictionary/prototype atom usage?

The project now compares four methods:

1. **Raw SVM**
   A linear SVM trained directly on the input pixels.

2. **Separate Dictionary + SVM**
   A two-stage baseline. First, a dictionary is trained using reconstruction
   and sparsity only. Then an SVM is trained on the fixed sparse codes.

3. **Separate Prototype + SVM**
   A two-stage prototype baseline. First, a dictionary is trained using the
   same simplex code constraint used by the prototype version of the joint
   model. Then an SVM is trained on the fixed prototype codes. This isolates
   the effect of prototype representation from the effect of joint
   optimization.

4. **Joint Prototype + SVM**
   The current proposed method. The dictionary, prototype codes, classifier,
   and auxiliary margin variables are optimized together in one coupled
   objective. The simplex constraint makes each code an interpretable mixture
   of learned prototypes.

The current implementation is not only a final model. It is also a diagnostic
framework for understanding when the joint formulation helps, when it overfits,
and which metrics are needed to support a rigorous thesis discussion.

The current empirical conclusion is cautious:

- the joint method is competitive on some harder Fashion-MNIST pairs;
- it is not uniformly better than the separate baseline;
- its main weakness is generalization, because training accuracy often reaches
  1.0 while validation/test margins and test accuracy do not improve
  consistently.

This is still useful for the thesis because it identifies the behavior of the
joint formulation rather than only reporting a single accuracy number.

## 2. Mathematical Formulation

### 2.1 Data And Variables

Let

$$
X = [x_1,\dots,x_n] \in [0,1]^{d \times n},
\qquad
y \in \{-1,+1\}^n.
$$

The variables are

$$
D \in [0,1]^{d \times m},
\qquad
C = [c_1,\dots,c_n] \in \mathbb{R}^{m \times n},
\qquad
w \in \mathbb{R}^m,
\qquad
b \in \mathbb{R},
\qquad
u \in \mathbb{R}^n.
$$

Their roles are:

- `D`: dictionary atoms;
- `C`: sparse codes;
- `w, b`: linear classifier in code space;
- `u`: auxiliary copy of the margin residual.

The classifier score for sample `j` is

$$
s_j = w^\top c_j + b.
$$

The margin residual is

$$
r_j(C,w,b) = 1 - y_j(w^\top c_j + b).
$$

If `r_j <= 0`, the sample satisfies the margin. If `r_j > 0`, the sample
violates the margin.

### 2.2 Original Hinge Objective

Before introducing the auxiliary variable, the intended objective is

$$
\min_{C,D,w,b}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,r_j(C,w,b))
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

The terms are:

- reconstruction loss: makes `D C` approximate `X`;
- classifier regularization: controls the size of `w`;
- hinge loss: penalizes margin violations;
- L1 sparsity: encourages sparse codes;
- box constraint: keeps dictionary atoms in `[0,1]`.

### 2.3 Auxiliary-Variable Reformulation

The implementation uses an auxiliary variable `u` to decouple the hinge term
from the classifier residual. Define

$$
q_j(C,w,b,u) = u_j - r_j(C,w,b).
$$

The implemented objective is

$$
\min_{C,D,w,b,u}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_{j=1}^n q_j(C,w,b,u)^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

Here:

- `rho` controls how strongly `u_j` is forced to approximate `r_j`;
- `eta` controls the hinge penalty applied to `u_j`;
- for large `rho`, the split formulation becomes closer to the original hinge
  formulation.

### 2.4 Important Interpretation Of `u`

`u` is **not** a nonnegative slack variable.

It is an auxiliary copy of the margin residual. Since

$$
r_j(C,w,b) = 1 - y_j(w^\top c_j + b)
$$

can be positive or negative, `u_j` must also be allowed to be positive or
negative.

Therefore:

- `u in R^n`;
- there is no constraint `u >= 0`;
- there is no indicator term `delta_{R_+^n}(u)`;
- `u` should not be described as a slack variable.

Correct wording:

> `u` is an auxiliary variable introduced to decouple the hinge term from the
> classifier residual. The quadratic penalty enforces `u_j ~= r_j(C,w,b)`, and
> the hinge penalty `max(0,u_j)` approximates `max(0,r_j(C,w,b))`.

### 2.5 Prototype-Learning Variant

After the initial joint sparse-coding formulation, a prototype-learning variant
was added to improve interpretability.

The key modeling change is to constrain each code vector to lie on the
probability simplex:

$$
c_j \in \Delta_m
= \left\{
c \in \mathbb{R}^m:
c_i \ge 0,\;
\sum_{i=1}^m c_i = 1
\right\}.
$$

Under this constraint, each sample is represented as a convex combination of
dictionary atoms:

$$
x_j \approx D c_j
= \sum_{i=1}^m c_{ij} d_i,
\qquad
c_{ij} \ge 0,
\qquad
\sum_i c_{ij}=1.
$$

This makes the dictionary atoms easier to interpret as prototypes:

> A sample can be explained as a weighted mixture of a small number of learned
> prototype atoms.

The prototype variant replaces the L1 code penalty with a simplex indicator:

$$
\delta_{\Delta_m^n}(C)
=
\begin{cases}
0, & c_j \in \Delta_m \text{ for all } j,\\
+\infty, & \text{otherwise}.
\end{cases}
$$

The resulting objective is

$$
\min_{C,D,w,b,u}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \lambda_w\|w\|_1
+ \frac{\rho}{2}\sum_{j=1}^n q_j(C,w,b,u)^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \delta_{\Delta_m^n}(C)
+ \delta_{[0,1]^{d \times m}}(D).
$$

Important note:

If `c_j >= 0` and `sum_i c_ij = 1`, then

$$
\|c_j\|_1 = 1.
$$

Therefore, the original L1 term `mu ||C||_1` becomes constant under the
simplex constraint and no longer controls sparsity. In the implementation, the
prototype option sets the code nonsmooth term to the simplex indicator instead
of the L1 penalty.

Practical sparsity is still measured using thresholds such as `1e-3` and
`1e-2`. In a simplex representation, high threshold sparsity means that each
sample is represented by only a few prototypes with meaningful positive mass.

### 2.6 Composite Proximal-Gradient Splitting

The problem is written as

$$
F(x) = f(x) + h(x),
\qquad
x = (C,D,w,b,u).
$$

The smooth part is

$$
f(C,D,w,b,u)
= \frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_{j=1}^n q_j(C,w,b,u)^2.
$$

The nonsmooth part is

$$
h(C,D,u)
= \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D).
$$

This splitting gives simple proximal updates for `C`, `D`, and `u`.

### 2.7 Proximal Operators

For `C`, the prox is soft thresholding:

$$
\operatorname{prox}_{t\mu\|\cdot\|_1}(Z)
= \operatorname{sign}(Z)\odot\max(|Z|-t\mu,0).
$$

For the prototype-learning variant, the prox for `C` is instead column-wise
projection onto the probability simplex:

$$
\operatorname{prox}_{\delta_{\Delta_m}}(z)
= \Pi_{\Delta_m}(z),
\qquad
\Delta_m = \{c: c_i \ge 0,\; \sum_i c_i = 1\}.
$$

This enforces `C >= 0` and `sum_i c_ij = 1` for every sample.

For `D`, the prox is projection onto the box constraint:

$$
\operatorname{prox}_{\delta_{[0,1]}}(Z) = \operatorname{clip}(Z,0,1).
$$

For `u`, because `u` is unconstrained and the nonsmooth term is

$$
\frac{\eta}{2}\max(0,u_j),
$$

the elementwise prox is

$$
\operatorname{prox}(z) =
\begin{cases}
z, & z < 0, \\
0, & 0 \le z \le t\eta/2, \\
z - t\eta/2, & z > t\eta/2.
\end{cases}
$$

This is different from projecting `u` to be nonnegative. Negative values are
kept because negative margin residuals represent samples that satisfy the
margin.

### 2.8 Backtracking Rule

The solver uses the composite proximal-gradient upper-bound condition. For a
trial point `x_trial`, the smooth part must satisfy

$$
f(x^{trial}) \le
f(x^k)
+ \langle \nabla f(x^k), x^{trial} - x^k \rangle
+ \frac{1}{2t}\|x^{trial} - x^k\|^2.
$$

This is the correct backtracking rule for the composite objective `F = f + h`.
It replaces the earlier plain sufficient decrease test on the full objective.

## 3. Repository Structure

The repository is organized as:

```text
FYP_05/
  FYP (2).pdf
  plan.md
  README.md
  requirements.txt
  src/
  notebooks/
  tests/
  fyp/
```

Main roles:

- `FYP (2).pdf`: original mathematical process and project reference.
- `plan.md`: this technical project plan and implementation record.
- `requirements.txt`: Python dependencies.
- `src/`: reusable implementation code.
- `notebooks/`: experiment workflows and result visualizations.
- `tests/`: unit and smoke tests.
- `fyp/`: local virtual environment.

## 4. Source Code Walkthrough

### 4.1 `src/config.py`

This file centralizes task definitions and hyperparameters.

Classes:

- `DataConfig`
  Stores the default binary dataset configuration: labels, split sizes,
  normalization, and random seed.

- `TaskConfig`
  Describes a single binary classification task. It stores dataset name,
  positive labels, negative labels, split sizes, and seed. It also provides
  `loader_kwargs()` so that the data loader can be called consistently.

- `HyperParams`
  Stores optimization and model hyperparameters:
  `dictionary_size`, `mu`, `code_simplex`, `rho`, `gamma`, `w_l1`, `eta`,
  initialization scales, backtracking settings, `max_iter`, `tol`, correction
  settings, and `random_state`.

Functions:

- `default_data_config()`
  Returns the default `DataConfig`.

- `default_hyperparams()`
  Returns the default `HyperParams`.

- `as_dict(config)`
  Converts a dataclass configuration into a dictionary.

- `report_task_suite()`
  Defines the compact MNIST binary task suite used in the main report.

- `mnist_task_suite()`
  Alias for the report MNIST task suite.

- `extended_task_suite()`
  Adds extra MNIST pairs for stress testing.

- `fashion_task_suite()`
  Defines harder Fashion-MNIST binary tasks.

- `one_vs_rest_suite()`
  Defines MNIST one-vs-rest tasks for broader supplementary analysis.

- `task_catalog()`
  Provides a dictionary mapping task names to `TaskConfig` objects.

### 4.2 `src/data.py`

This file handles dataset loading and binary task construction.

Important functions:

- `_load_local_mnist_npz(path)`
  Loads MNIST from a local `.npz` file if available.

- `_load_local_fashion_mnist_npz(path)`
  Loads Fashion-MNIST from a local `.npz` file if available.

- `_load_mnist_arrays()`
  Loads MNIST arrays.

- `_load_fashion_mnist_arrays()`
  Loads Fashion-MNIST arrays. The code supports local data and fallback
  loaders.

- `load_binary_task(...)`
  General binary task loader. It selects positive and negative labels, maps
  them to `+1` and `-1`, normalizes images to `[0,1]`, and returns column-major
  matrices.

- `load_mnist_binary(...)`
  Backward-compatible helper for the original MNIST binary task.

- `load_task(task)`
  Loads data directly from a `TaskConfig`.

Data output format:

```text
X_train: d x n_train
y_train: n_train
X_val:   d x n_val
y_val:   n_val
X_test:  d x n_test
y_test:  n_test
```

### 4.3 `src/model.py`

This file defines the objective and gradients.

Functions:

- `margin_residual(C, w, b, y)`
  Computes `r_j = 1 - y_j(w^T c_j + b)`.

- `penalty_residual_q(C, w, b, u, y)`
  Computes `q_j = u_j - r_j(C,w,b)`.

- `smooth_objective(params, X, y, hyper)`
  Computes the smooth objective components:
  reconstruction, classifier regularization, and quadratic penalty.

- `nonsmooth_objective(params, hyper)`
  Computes the nonsmooth objective components:
  hinge term on `u`, L1 sparsity on `C`, and the dictionary indicator.

- `objective(params, X, y, hyper)`
  Combines smooth and nonsmooth components into the total objective.

- `gradients(params, X, y, hyper)`
  Computes gradients of the smooth part with respect to `C`, `D`, `w`, `b`,
  and `u`.

### 4.4 `src/prox.py`

This file contains the proximal operators.

Functions:

- `prox_C(C_tilde, step, mu, simplex=False)`
  Soft-thresholding for the L1 code penalty in the original sparse-coding
  model. If `simplex=True`, it projects each column onto the probability
  simplex for prototype learning.

- `project_columns_to_simplex(Z)`
  Projects each code column onto `{c >= 0, sum(c) = 1}`.

- `prox_w(w_tilde, step, w_l1)`
  Soft-thresholding for the optional sparse SVM head.

- `prox_D(D_tilde)`
  Clips dictionary entries to `[0,1]`.

- `prox_u(u_tilde, step, eta)`
  Applies the correct unconstrained hinge prox for
  `(eta/2) * max(0,u)`. Negative `u` values are not clipped.

### 4.5 `src/init.py`

This file initializes joint model variables.

Function:

- `initialize_params(X, y, dictionary_size, seed, code_scale, classifier_scale)`
  Initializes:
  - `D` from random training columns;
  - `C` with small random values;
  - `w` with small random values;
  - `b` as zero;
  - `u` as zero.

Small nonzero initialization is used because zero classifier/code
initialization can delay or weaken the classification branch.

### 4.6 `src/solver.py`

This file implements the joint proximal-gradient solver.

Functions:

- `_copy_params(params)`
  Copies parameter dictionaries safely.

- `_flatten_diff_sq(old, new)`
  Computes squared Euclidean distance between parameter dictionaries.

- `_flatten_inner_product(left, right)`
  Computes inner products between gradient and update directions.

- `gradient_step(params, grads, step)`
  Applies the explicit gradient step for the smooth part.

- `prox_step(trial, step, hyper)`
  Applies the proximal updates for `C`, `D`, and `u`.

- `classification_diagnostics(params, y)`
  Records training diagnostics such as score gap, violation rate, positive
  violation, `w` norm, and `u-r` mismatch.

- `fit_joint_pg(X, y, hyper, init_params)`
  Runs the joint proximal-gradient algorithm with backtracking. It returns:
  - final parameters;
  - objective history;
  - component history;
  - diagnostic history;
  - status.

### 4.7 `src/baselines.py`

This file implements the baseline methods.

Functions:

- `fit_raw_svm(X_train, y_train, X_test, y_test, hyper)`
  Trains a linear SVM directly on pixels.

- `fit_separate_dictionary(X_train, hyper, init_params)`
  Trains a dictionary using reconstruction and L1 sparsity only.

- `_encode_with_fixed_dictionary(...)`
  Infers sparse codes for new samples using a fixed dictionary.

- `fit_separate_dict_svm(X_train, y_train, X_test, y_test, hyper)`
  Runs the two-stage baseline: train dictionary, infer codes, then train SVM
  on the codes.

### 4.8 `src/metrics.py`

This file contains evaluation and diagnostic metrics.

Core prediction and reconstruction:

- `predict_from_codes(C, w, b)`
- `accuracy_from_codes(C, y, w, b)`
- `reconstruction_error(X, D, C)`

Sparsity:

- `code_sparsity(C, threshold)`
- `code_sparsity_summary(C)`

`code_sparsity_summary` reports multiple thresholds:

- `code_sparsity`: exact or near-exact zero using `1e-10`;
- `code_sparsity_1em4`: practical sparsity at `1e-4`;
- `code_sparsity_1em3`: practical sparsity at `1e-3`;
- `code_sparsity_1em2`: loose sparsity at `1e-2`.

Decision and margin diagnostics:

- `decision_statistics_from_scores(scores, y)`
- `decision_statistics(C, y, w, b)`

These report:

- score mean and standard deviation;
- positive and negative class score means;
- `score_gap` as an optional internal diagnostic;
- margin residual;
- positive margin violation;
- `violation_rate`;
- `margin_satisfaction_rate`.

Model evaluation:

- `infer_codes_with_dictionary(...)`
  Infers sparse codes for validation/test data using a fixed dictionary.

- `evaluate_joint_model(...)`
  Evaluates a trained joint model on validation/test data.

- `summarize_joint_result(...)`
  Summarizes the training split for a joint result.

Diagnostics:

- `joint_code_distribution_report(...)`
  Compares train/validation/test code distributions.

- `overfitting_diagnostic_summary(...)`
  Summarizes train-test gap, score-gap retention, violation gaps, and `w` norm.

- `joint_component_scale_report(...)`
  Reports how much each objective component contributes.

- `format_training_diagnostic_trajectory(...)`
  Prints selected iterations of the training trajectory.

- `run_joint_sensitivity_scan(...)`
  Runs small hyperparameter scans.

### 4.9 `src/experiments.py`

This file orchestrates multi-task experiments.

Functions:

- `_svm_split_summary(...)`
  Summarizes a raw SVM split.

- `_code_split_summary(...)`
  Summarizes an SVM trained on sparse codes.

- `fit_svm_on_fixed_dictionary(...)`
  Diagnostic helper that trains an SVM using a fixed dictionary.

- `joint_dictionary_svm_diagnostic(...)`
  Diagnostic helper that evaluates whether the joint dictionary itself is
  useful when paired with a separately trained SVM. This is diagnostic only and
  is not part of the main result table.

- `benchmark_binary_task(task, baseline_hyper, joint_hyper)`
  Runs Raw SVM, Separate Dictionary + SVM, optional Separate Prototype + SVM,
  and Joint Dictionary + SVM on one binary task. The Separate Prototype + SVM
  row is included when `joint_hyper.code_simplex=True`.

- `run_task_suite(tasks, baseline_hyper, joint_hyper)`
  Runs a list of binary tasks.

- `flatten_comparison_rows(task_results)`
  Converts nested task results into rows suitable for tables.

- `summarize_method_aggregate(rows)`
  Aggregates results by method.

- `format_method_aggregate_summary(summary_rows)`
  Renders aggregate result tables.

## 5. Notebook Workflow

The notebooks are used for experimentation and presentation.

- `notebooks/01_problem_setup.ipynb`
  Initial problem setup and intuition.

- `notebooks/02_baselines.ipynb`
  Runs the Raw SVM and Separate Dictionary + SVM baselines.

- `notebooks/03_joint_optimization.ipynb`
  Main joint optimization notebook. It is used for training diagnostics,
  sensitivity scans, and checking the behavior of the joint solver.

- `notebooks/04_results_analysis.ipynb`
  MNIST result analysis.

- `notebooks/05_fashion_results.ipynb`
  Fashion-MNIST result analysis.

- `notebooks/06_fashion_joint_optimization.ipynb`
  Fashion-MNIST joint optimization and diagnostic analysis.

- `notebooks/07_multi_pair_results.ipynb`
  Current meeting-ready notebook. It runs multiple MNIST and Fashion-MNIST
  binary pairs and compares Raw SVM, Separate Dictionary + SVM, Separate
  Prototype + SVM, and Joint Dictionary + SVM using fixed hyperparameters.

## 6. Data Flow

The full experimental data flow is:

```text
TaskConfig
  -> load_task
  -> X_train, y_train, X_val, y_val, X_test, y_test
  -> baseline_hyper / joint_hyper
  -> Raw SVM
  -> Separate Dictionary + SVM
  -> Separate Prototype + SVM when code_simplex=True
  -> Joint Dictionary + SVM
  -> metrics
  -> comparison rows
  -> aggregate tables and plots
```

Method-specific data flow:

### Raw SVM

```text
X_train, y_train
  -> LinearSVC on pixels
  -> train/val/test accuracy and margin diagnostics
```

### Separate Dictionary + SVM

```text
X_train
  -> reconstruction-only dictionary learning
  -> D and C_train
  -> infer C_val and C_test using fixed D
  -> LinearSVC on sparse codes
  -> accuracy, margin, reconstruction, sparsity
```

### Separate Prototype + SVM

```text
X_train
  -> reconstruction-only dictionary learning with c_j >= 0 and 1^T c_j = 1
  -> D and prototype-mixture C_train
  -> infer C_val and C_test using fixed D and the same simplex constraint
  -> LinearSVC on prototype codes
  -> accuracy, margin, reconstruction, sparsity, prototype interpretability
```

This baseline is important for control-variable reasoning. If Joint Prototype
outperforms Separate Dictionary but not Separate Prototype, the improvement is
mainly due to the prototype representation. If Joint Prototype outperforms
Separate Prototype, the additional gain is attributable to the joint
classification-reconstruction coupling.

### Joint Prototype + SVM

```text
X_train, y_train
  -> initialize C, D, w, b, u
  -> proximal-gradient updates of all variables
  -> final D, C_train, w, b, u
  -> infer C_val and C_test using fixed D and the simplex code constraint
  -> evaluate classifier w,b on inferred codes
  -> accuracy, margin, reconstruction, sparsity, explainability diagnostics
```

The key comparison is `Separate Prototype + SVM` versus
`Joint Prototype + SVM`. Both use the same prototype/simplex representation,
so differences between them are evidence for or against the additional value
of joint classification-reconstruction optimization.

## 7. Metrics And Interpretation

### Accuracy Metrics

- `train_accuracy`
  Accuracy on the training split.

- `val_accuracy`
  Accuracy on the validation split.

- `test_accuracy`
  Accuracy on the test split.

- `train_test_gap`
  Difference between train and test accuracy. A large gap indicates
  overfitting.

### Margin Metrics

- `score_gap` optional diagnostic
  Difference between the mean positive-class score and mean negative-class
  score:

  ```text
  score_gap = mean(score | y=+1) - mean(score | y=-1)
  ```

  A larger score gap can indicate stronger class separation, but it is not used
  as a primary result metric in the current report because it is less stable
  across methods. It remains available for debugging score distributions.

- `violation_rate`
  Fraction of samples with positive margin residual:

  ```text
  violation_rate = mean(r_j > 0)
  ```

  Lower is better. A high violation rate means many samples do not satisfy the
  unit-margin condition, even if their predicted signs may still be correct.

- `mean_positive_violation`
  Average hinge violation:

  ```text
  mean(max(0, r_j))
  ```

  Lower is better.

- `margin_satisfaction_rate`
  Fraction of samples with `r_j <= 0`. Higher is better.

### Reconstruction Metrics

- `reconstruction_error`
  Frobenius norm:

  ```text
  ||X - D C||_F
  ```

  Lower means better reconstruction. However, better reconstruction does not
  necessarily imply better classification.

### Sparsity Metrics

- `code_sparsity`
  Fraction of code entries close to zero using threshold `1e-10`.

- `code_sparsity_1em4`
  Fraction of code entries with absolute value at most `1e-4`.

- `code_sparsity_1em3`
  Fraction of code entries with absolute value at most `1e-3`.

- `code_sparsity_1em2`
  Fraction of code entries with absolute value at most `1e-2`.

The practical sparsity thresholds are useful because proximal-gradient codes
may contain very small nonzero numerical values.

### Metric Groups For The Final Experimental Claims

The final report uses metrics according to three claims.

**Representative features**

- `test_reconstruction_error`: whether the learned dictionary/prototypes can
  reconstruct the input.
- `test_code_sparsity_1em3`: whether the representation is compact.
- top activated atoms/prototypes: whether the representation can be inspected
  visually.

Representative quality is therefore not defined by sparsity alone. A useful
representation should reconstruct the sample reasonably well while using a
compact set of atoms.

**Discriminative features**

- `test_accuracy`: whether predictions are correct.
- `train_test_gap`: whether the classifier generalizes.
- `test_violation_rate`: whether samples satisfy the SVM margin.
- SVM margin projection: visual evidence of boundary and margin behavior.

The main discriminative margin evidence is `violation_rate`, not `score_gap`.
Lower violation rate means more samples lie outside the unit-margin band on the
correct side.

**Explainability**

- prototype coefficients: which atoms are used by a sample;
- top activated atoms: what visual patterns the sample is reconstructed from;
- atom contribution `w_i c_i`: which atoms push the classifier toward the
  positive or negative class;
- margin projection: whether the sample is confidently classified, close to
  the boundary, or misclassified.

### Objective Component Metrics

The component scale report tracks:

- reconstruction term;
- quadratic penalty term;
- hinge term;
- L1 term;
- classifier regularization term.

This diagnostic is used to check whether one part of the objective dominates
the optimization. Earlier experiments showed that reconstruction can dominate
the total objective numerically. A dimension-normalized objective scaling was
tested, but it made optimization and accuracy worse, so it was rolled back.

### Overfitting Diagnostics

The overfitting diagnostic tracks:

- train-validation accuracy gap;
- train-test accuracy gap;
- test margin violation gap;
- optional score-gap retention from train to test;
- test/train code scale ratio;
- `w_norm`.

Current joint results show a recurring pattern:

- training accuracy often reaches 1.0;
- validation/test accuracy does not improve consistently;
- test violation rate remains high;
- `w_norm` can become large.

This means the classifier branch learns the training set but does not
generalize reliably.

## 8. Hyperparameters

The main hyperparameters are:

- `dictionary_size`
  Number of dictionary atoms. Larger values increase representation capacity
  but may increase overfitting.

- `mu`
  L1 sparsity weight for codes. Larger values make codes sparser but may harm
  reconstruction or classification if too strong. In the prototype-learning
  variant with `code_simplex=True`, this term is not active for `C` because
  each column has fixed L1 norm equal to one.

- `code_simplex`
  Enables the prototype-learning variant. If true, each code vector is
  projected onto the probability simplex, enforcing nonnegative coefficients
  that sum to one.

- `gamma`
  L2 regularization weight on `w`. Larger values reduce classifier freedom and
  can reduce overfitting.

- `w_l1`
  Optional L1 penalty on the classifier weights. This produces a sparse SVM
  head and makes the classifier depend on fewer dictionary atoms.

- `rho`
  Quadratic penalty weight enforcing `u ~= r`. Larger values force the
  auxiliary variable to match the margin residual more closely.

- `eta`
  Hinge penalty weight on `u`. Larger values emphasize classification margin
  violations, but may also overfit the training set.

- `init_code_scale`
  Scale of the random initialization of `C`.

- `init_classifier_scale`
  Scale of the random initialization of `w`.

- `initial_step`
  Initial proximal-gradient step size.

- `backtracking_shrink`
  Factor used to shrink the step size during backtracking.

- `backtracking_min_step`
  Minimum allowed step size.

- `max_iter`
  Maximum number of proximal-gradient iterations.

- `tol`
  Relative objective change tolerance.

- `random_state`
  Random seed for reproducibility.

Current fixed diagnostic configuration used in multi-pair experiments:

```python
dictionary_size = 48
mu = 0.3
code_simplex = False
gamma = 0.5
w_l1 = 0.0
rho = 10
eta = 10
init_code_scale = 5e-3
init_classifier_scale = 5e-3
max_iter = 100
random_state = 7
```

This is not claimed to be globally optimal. It is a stable fixed configuration
for diagnostic comparison.

For prototype-learning experiments, the key switch is:

```python
code_simplex = True
```

Then each sample is interpreted as a convex combination of learned prototypes.

## 9. Experimental Attempts And Improvements

### 9.1 Initial MNIST 3 vs 8 Experiment

The project began with MNIST `3 vs 8`.

Initial observations:

- Raw SVM was already strong.
- Separate Dictionary + SVM was competitive.
- Joint Dictionary + SVM did not consistently outperform the separate
  baseline.
- The objective was dominated by reconstruction scale.

This showed that MNIST `3 vs 8` alone was not sufficient to demonstrate a clear
joint-learning advantage.

### 9.2 Correcting The Auxiliary Variable Formulation

The original implementation treated `u` like a nonnegative slack variable.
This was mathematically inconsistent because `u` is intended to approximate the
margin residual, which can be negative.

Corrections made:

- removed the nonnegativity interpretation of `u`;
- removed any implicit `u >= 0` projection;
- corrected `prox_u` for the unconstrained hinge term;
- updated comments and documentation to describe `u` as an auxiliary residual
  copy.

Effect:

- the code became consistent with the teacher-confirmed formulation;
- negative margin residuals are now handled correctly.

### 9.3 Correcting Backtracking

The earlier line search used a plain sufficient decrease test on the full
objective. This is not the standard condition for composite proximal-gradient
methods.

Correction made:

- implemented the smooth-part quadratic upper-bound condition.

Effect:

- the solver now matches the proximal-gradient splitting used in the
  mathematical formulation.

### 9.4 Adding Self-Checks

Each `src/*.py` file was given a small `if __name__ == "__main__":` self-check
entry where appropriate.

Purpose:

- allow direct file-level sanity checks;
- make debugging easier;
- provide lightweight verification outside notebooks.

### 9.5 Hyperparameter Scans

Several hyperparameter directions were tested:

- `rho` and `eta` to control the classification/coupling strength;
- `dictionary_size` and `mu` to control representation capacity and sparsity;
- `gamma` and `max_iter` to control overfitting;
- initialization scales for `C` and `w`.

Observed effects:

- increasing classification strength can improve training margins;
- stronger sparsity can increase practical code sparsity;
- larger `gamma` can reduce classifier freedom;
- longer training sometimes worsens generalization;
- no single hyperparameter setting made joint uniformly better than separate.

### 9.6 Objective Scaling Attempt

A dimension-normalized objective scaling was tested because reconstruction sums
over `d x n`, while classification terms sum mainly over `n`.

Observed effect:

- the scaled objective made optimization slower and accuracy worse;
- backtracking often accepted step size 1 but the useful learning behavior was
  weaker;
- this approach was rolled back.

Conclusion:

- objective scaling is a reasonable theoretical concern, but the tested
  normalization was not beneficial in this implementation.

### 9.7 Fashion-MNIST Extension

Fashion-MNIST was added because MNIST pairs are often too easy for Raw SVM and
Separate Dictionary + SVM.

Fashion-MNIST task suite:

- T-shirt vs Shirt;
- Pullover vs Coat;
- Dress vs Coat;
- Sandal vs Sneaker.

Observed effects:

- Fashion-MNIST provides harder representation-learning tasks;
- joint is competitive and sometimes better on selected pairs;
- the method still shows generalization problems.

### 9.8 Additional Metrics

The following diagnostics were added:

- multiple sparsity thresholds;
- score gap;
- violation rate;
- mean positive violation;
- objective component scale report;
- training diagnostic trajectory;
- code distribution mismatch report;
- overfitting diagnostic summary;
- multi-pair aggregate result tables.

Effect:

- the analysis moved beyond test accuracy alone;
- the current weakness of joint learning became clearer: training classification
  succeeds, but validation/test margins often remain weak.

### 9.9 Multi-Pair Result Notebook

`notebooks/07_multi_pair_results.ipynb` was created as a meeting-ready
diagnostic and report notebook.

It compares Raw SVM, Separate Dictionary + SVM, Separate Prototype + SVM, and
Joint Prototype + SVM across MNIST and Fashion-MNIST pairs using fixed
hyperparameters.

The report structure is:

1. **Task-level comparison table**
   Reports test accuracy, train-test gap, margin violation rate,
   reconstruction error, and practical code sparsity.

2. **Standalone margin violation table**
   Compares `test_violation_rate` across all methods on each pair. Lower is
   better because more samples satisfy the SVM margin. This table supports the
   discriminative-feature discussion without mixing margin evidence with
   reconstruction or sparsity.

3. **Standalone sparsity table**
   Compares `test_code_sparsity_1em3` across dictionary-based methods. Higher
   is better because more coefficients are practically zero. Raw SVM is
   excluded because it has no learned code representation.

4. **Dataset/method aggregate table**
   Summarizes mean test accuracy, train-test gap, violation rate,
   reconstruction error, and sparsity separately for MNIST and Fashion-MNIST.

5. **Main plots**
   The main visual results are:
   - test accuracy by task;
   - train-test accuracy gap by task;
   - test margin violation rate by task;
   - reconstruction-vs-accuracy trade-off;
   - sparsity-vs-accuracy trade-off.

The Separate Prototype + SVM row was added after the prototype-learning
extension. Its purpose is to avoid attributing all improvements to joint
optimization when they may instead come from the nonnegative simplex
representation itself.

The intended interpretation is:

- `Separate Dict + SVM` tests whether reconstruction-only dictionary learning
  provides useful features.
- `Separate Prototype + SVM` tests whether the simplex prototype constraint
  alone improves compactness and explainability.
- `Joint Prototype + SVM` tests whether adding classification feedback during
  representation learning improves discriminative behavior compared with
  separate prototype learning.

### 9.10 Explainability Visualizations

Explainability tools were added to make the learned representation easier to
inspect.

The notebook now runs explainability on two representative tasks:

- Fashion-MNIST `T-shirt vs Shirt`, because it is visually harder and more
  relevant for representation learning;
- MNIST `3 vs 8`, because digit shapes are easier to interpret visually.

For each selected task, the analysis includes:

- SVM-style margin projection with decision boundary and two margin lines;
- side-by-side margin views for Raw SVM, Separate Dictionary + SVM, Separate
  Prototype + SVM, and Joint Prototype + SVM;
- representative correct, margin-violating, and misclassified examples;
- original image, reconstruction, and residual image;
- top activated atoms for a sample;
- atom-level classifier contribution `w_i c_i`.

Purpose:

- show not only whether a sample is classified correctly, but also why it is
  close to or far from the classifier boundary;
- connect representation learning to classification through atom
  contributions;
- explain correct and incorrect predictions through the prototypes that are
  activated and the sign/magnitude of their classifier contributions.

### 9.11 Sparse SVM Head

A sparse classifier head was added through an optional L1 penalty on `w`:

$$
\lambda_w\|w\|_1.
$$

This is implemented by soft-thresholding `w` in the proximal step.

Purpose:

- reduce classifier overfitting;
- make the classifier depend on fewer dictionary atoms;
- improve interpretability by identifying discriminative atoms.

Initial experiments showed that a small or moderate `w_l1` can slightly reduce
classifier density, but it did not by itself solve the generalization issue.

### 9.12 Prototype-Learning Variant

The prototype-learning variant was added after feedback that the representation
should be more explainable.

Implementation:

- set `code_simplex=True`;
- replace the code L1 prox with column-wise simplex projection;
- enforce `C >= 0` and `sum_i c_ij = 1`;
- keep `D` constrained to `[0,1]`.

Interpretation:

> Each input is represented as a convex combination of learned dictionary
> prototypes.

Initial quick check on Fashion-MNIST `T-shirt vs Shirt` confirmed:

- training and test codes satisfy nonnegativity and column-sum-one constraints;
- practical code sparsity under thresholds such as `1e-3` and `1e-2` becomes
  much higher;
- test accuracy remains in the same broad range but should be evaluated more
  systematically across pairs.

## 10. Current Empirical Findings

The current result pattern is:

- MNIST is often too easy for demonstrating a joint advantage.
- Raw SVM and Separate Dictionary + SVM are already strong on many MNIST pairs.
- Joint often reaches training accuracy 1.0, so the classifier branch is able
  to fit the training data.
- Joint does not consistently improve validation/test accuracy.
- Joint often has larger train-test gap than Separate Dictionary + SVM.
- Joint often has higher test violation rate, meaning more test samples fail
  the unit-margin condition.
- Fashion-MNIST gives a better setting for observing potential representation
  benefits, especially for harder pairs such as T-shirt vs Shirt and Dress vs
  Coat.

The most defensible current conclusion is:

> The joint formulation is promising and can be competitive on harder
> representation-learning tasks, but the current implementation does not yet
> show uniform superiority over the separate baseline. The main unresolved
> issue is generalization rather than training optimization failure.

## 11. Why Overfitting Happens In The Current Results

The overfitting diagnosis is based on the observed metrics:

1. Training accuracy reaches 1.0.
   The joint classifier can fit the training codes very well.

2. Test accuracy does not improve consistently.
   The learned training separation does not transfer reliably to validation or
   test codes.

3. Test violation rate remains high.
   Many test samples are classified with insufficient margin, even when the
   predicted sign may be correct.

4. Score gap is lower on test than train.
   The separation learned on training codes is weaker on unseen inferred codes.

5. The train/test sparse-code distributions are not identical.
   Training codes are optimized jointly with the classifier, while validation
   and test codes are inferred later using the fixed dictionary. This creates a
   possible mismatch between optimized training codes and inferred test codes.

6. Classifier norm can become large.
   A large `w_norm` indicates that the classifier may be using too much
   capacity to separate training codes.

Therefore, the current problem is best described as:

> The joint method learns a discriminative representation for the training set,
> but the learned classifier/representation pair does not generalize as
> reliably as the separate baseline.

Potential remedies:

- stronger classifier regularization by increasing `gamma`;
- validation-based early stopping;
- smaller dictionary size;
- stronger sparsity through larger `mu`;
- smaller or more carefully tuned `eta`;
- smaller or more carefully tuned `rho`;
- more systematic validation-based hyperparameter selection;
- focusing analysis on harder but not impossible Fashion-MNIST pairs.

## 12. Testing And Verification

The project uses Python unit tests in `tests/`.

Run all tests with:

```bash
source fyp/bin/activate
python3 -m unittest discover -s tests -p 'test*.py' -v
```

Test files:

- `tests/test_smoke.py`
  Basic smoke tests for importability and simple end-to-end behavior.

- `tests/test_baselines.py`
  Checks baseline dictionary/SVM routines.

- `tests/test_solver.py`
  Checks solver behavior, proximal updates, and objective consistency.

- `tests/test_experiments.py`
  Checks task-suite experiment helpers and comparison row generation.

Additional verification:

- each major `src/*.py` file has a small self-check entry;
- notebook outputs are used for empirical diagnostics;
- objective component reports verify which terms dominate the optimization;
- training trajectory reports verify whether classification improves during
  training or collapses.

## 13. Recommended Next Steps

The next stage should avoid random hyperparameter guessing.

Recommended direction:

1. Keep the current multi-pair notebook for meeting discussion.

2. Use Fashion-MNIST as the main harder dataset, because MNIST is often too
   easy for showing representation-learning benefits.

3. Use validation-based model selection if tuning is resumed.

4. Prioritize generalization diagnostics:
   - test accuracy;
   - train-test gap;
   - test violation rate;
   - mean positive violation;
   - score-gap retention;
   - sparsity at `1e-3`;
   - reconstruction-vs-accuracy trade-off.

5. Consider validation early stopping, because longer optimization can improve
   training fitting while worsening validation/test behavior.

6. If the joint method still does not consistently outperform the separate
   baseline, present the thesis conclusion honestly:

   > The auxiliary-variable joint formulation is mathematically valid and
   > competitive on selected harder tasks, but under the current implementation
   > and fixed hyperparameter setting it does not uniformly dominate separate
   > dictionary learning. The main limitation is generalization.
