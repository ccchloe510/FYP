# Joint Dictionary Learning Repo Walkthrough Notes

## Purpose of This Note

This note is designed to help understand the repo at two levels:

- the **global project flow**
- the **responsibility of each file and function**

It is written so it can be:

- read directly inside the repo
- imported into Notion as a structured study note

The recommended reading strategy is:

1. understand the overall flow
2. understand the mathematical variables and shapes
3. understand how each file fits into the flow
4. only then read individual functions in detail

---

# 1. Project Goal

This repo implements a **joint optimization model** that combines:

- dictionary learning
- sparse coding
- linear SVM-style classification

inside **one coupled objective**

The main current experimental setting is:

- dataset: MNIST
- task type: binary classification
- default example: `3 vs 8`
- data normalization: `[0,1]`
- dictionary constraint: `D in [0,1]^{d x m}`
- optimizer: proximal gradient with backtracking

The main research question is:

> Does joint learning of representation and classifier give a better discriminative representation than a separate two-stage pipeline?

---

# 2. Why This Problem Is Joint

The model is called **joint** because the representation variables and the classifier variables are optimized together in the same objective.

The key variables are:

- `D`: dictionary
- `C`: sparse code matrix
- `w, b`: classifier parameters
- `u`: auxiliary variable

The reconstruction part depends on `D` and `C`:

\[
\frac{1}{2}\|X - DC\|_F^2
\]

The classification part depends on `C`, `w`, and `b` through:

\[
r_j = 1 - y_j (w^\top c_j + b)
\]

and the auxiliary-variable coupling:

\[
q_j = u_j - r_j
\]

So:

- `C` is not learned only for reconstruction
- the classifier directly influences the learned representation
- the learned representation also influences classifier updates

This is different from a separate pipeline:

- **Separate Dictionary + SVM**
  - first learn `D` and `C`
  - then fix `C`
  - then train `w, b`

- **Joint Dictionary + SVM**
  - optimize `D, C, w, b, u` together

---

# 3. Global Repo Flow

This is the most important high-level picture.

```text
Task definition
-> data loading and preprocessing
-> parameter initialization
-> objective definition
-> gradient computation
-> proximal operators
-> proximal-gradient solver
-> evaluation metrics
-> baseline comparison
-> results analysis
```

Mapped to files:

- task definition -> `src/config.py`
- data loading -> `src/data.py`
- initialization -> `src/init.py`
- objective and gradients -> `src/model.py`
- proximal operators -> `src/prox.py`
- joint solver -> `src/solver.py`
- baselines -> `src/baselines.py`
- metrics and evaluation -> `src/metrics.py`
- experiments and plots -> `notebooks/*.ipynb`

---

# 4. Main Variable Flow

This is the core runtime flow of the joint method.

```text
X, y
-> initialize D, C, w, b, u
-> compute residual r_j
-> compute penalty residual q_j
-> compute smooth objective f
-> compute nonsmooth objective h
-> compute gradients of f
-> gradient step
-> proximal step
-> update parameters
-> record objective history
-> evaluate accuracy / reconstruction / sparsity
```

For the separate baseline:

```text
X, y
-> learn dictionary D and codes C from reconstruction objective only
-> train linear SVM on C
-> evaluate accuracy
```

For raw SVM:

```text
X, y
-> train linear SVM on original features
-> evaluate accuracy
```

---

# 5. Core Mathematics

## 5.1 Data and Variables

\[
X = [x_1, \dots, x_n] \in [0,1]^{d \times n}
\]

\[
y \in \{-1,+1\}^n
\]

\[
D \in [0,1]^{d \times m}, \quad
C \in \mathbb{R}^{m \times n}, \quad
w \in \mathbb{R}^{m}, \quad
b \in \mathbb{R}, \quad
u \in \mathbb{R}^{n}
\]

## 5.2 Key Definitions

\[
r_j = 1 - y_j(w^\top c_j + b)
\]

\[
q_j = u_j - r_j
\]

\[
\|C\|_1 = \sum_{i,j} |c_{ij}|
\]

## 5.3 Objective

\[
\min_{C,D,w,b,u}
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_{j=1}^n q_j^2
+ \frac{\eta}{2}\sum_{j=1}^n \max(0,u_j)
+ \mu\|C\|_1
+ \delta_{[0,1]^{d \times m}}(D)
+ \delta_{\mathbb{R}_+^n}(u)
\]

## 5.4 Smooth/Nonsmooth Split

Smooth part:

\[
f =
\frac{1}{2}\|X - DC\|_F^2
+ \frac{\gamma}{2}\|w\|_2^2
+ \frac{\rho}{2}\sum_j q_j^2
\]

Nonsmooth part:

\[
h =
\mu\|C\|_1
+ \frac{\eta}{2}\sum_j \max(0,u_j)
+ \delta_{[0,1]}(D)
+ \delta_{\mathbb{R}_+^n}(u)
\]

## 5.5 Gradients

Let

\[
E = DC - X, \quad
q = [q_1,\dots,q_n]^\top, \quad
s = q \odot y
\]

Then:

\[
\nabla_C f = D^\top E + \rho w s^\top
\]

\[
\nabla_D f = E C^\top
\]

\[
\nabla_w f = \gamma w + \rho C s
\]

\[
\nabla_b f = \rho \mathbf{1}^\top s
\]

\[
\nabla_u f = \rho q
\]

## 5.6 Proximal Operators

\[
\operatorname{prox}_{\alpha \mu \|\cdot\|_1}(\widetilde C)
= \operatorname{sign}(\widetilde C)\odot\max(|\widetilde C|-\alpha\mu,0)
\]

\[
\operatorname{prox}(\widetilde D)=\operatorname{clip}(\widetilde D,0,1)
\]

\[
\operatorname{prox}(\widetilde u)=\max\left(0,\widetilde u-\alpha \frac{\eta}{2}\right)
\]

## 5.7 Solver Update

\[
\widetilde z^k = z^k - \alpha_k \nabla f(z^k)
\]

\[
z^{k+1} = \operatorname{prox}_{\alpha_k h}(\widetilde z^k)
\]

---

# 6. Important Shapes

This section is critical.

- `X`: `(d, n)`
- `D`: `(d, m)`
- `C`: `(m, n)`
- `w`: `(m,)`
- `b`: scalar
- `u`: `(n,)`
- `D @ C`: `(d, n)`
- `w @ C`: `(n,)`

This repo uses the **column-major sample convention**:

- one column = one sample

That is different from standard sklearn conventions, where one row is one sample.

---

# 7. File-by-File Walkthrough

## 7.1 `src/config.py`

### Purpose

Store default experiment settings in one place.

### Main Responsibilities

- define the binary task labels
- define split sizes
- define hyperparameters
- provide default config objects

### Important Objects

- `DataConfig`
- `HyperParams`
- `default_data_config()`
- `default_hyperparams()`

### In the Flow

This is the first step.

It defines:

- what task is being solved
- how much data to use
- what optimization settings to start from

### What To Notice

- `positive_labels` and `negative_labels` are now general, not hardcoded to one pair of digits
- this makes later tasks like `1 vs 9` or `1 vs rest` possible without rewriting the loader

---

## 7.2 `src/data.py`

### Purpose

Load MNIST and convert it into the binary task format required by the joint model.

### Main Responsibilities

- load raw MNIST arrays
- support local `.npz` loading and OpenML fallback
- define a binary classification task
- remap labels into `{-1,+1}`
- split into train / val / test
- normalize inputs to `[0,1]`
- transpose data into `(d, n)`

### Main Function

- `load_mnist_binary(...)`

### In the Flow

This file turns raw image classification data into model-ready tensors.

### Key Logic

1. load full MNIST
2. select positive and negative labels
3. map positive labels to `+1`, negative labels to `-1`
4. sample the requested total number of examples
5. split into train/val/test with stratification
6. normalize to `[0,1]`
7. transpose to `(d, n)`

### Why It Matters

Without this file:

- labels would not match the SVM margin formula
- the data scale would not match the dictionary constraint
- `X` would not have the right shape for `D @ C`

---

## 7.3 `src/init.py`

### Purpose

Initialize the optimization variables `D, C, w, b, u`.

### Main Responsibilities

- initialize `D` from actual training samples
- initialize `C` to zero
- initialize `w, b` to zero
- initialize `u` from the margin residual

### In the Flow

This file takes `X_train, y_train` and creates the initial point for optimization.

### Why The Initialization Looks Like This

- `D` is initialized from training examples so the atoms already lie in the data manifold and satisfy the box constraint
- `C` sparsity is learned later, so zero initialization is acceptable
- `w, b` are simple classifier initialization
- `u = max(0, r)` matches the nonnegative auxiliary-variable interpretation

---

## 7.4 `src/model.py`

### Purpose

Define the mathematical model in code.

### Main Responsibilities

- compute `r_j`
- compute `q_j`
- compute smooth objective terms
- compute nonsmooth objective terms
- compute total objective
- compute gradients

### Main Functions

- `margin_residual`
- `penalty_residual_q`
- `smooth_objective`
- `nonsmooth_objective`
- `objective`
- `gradients`

### In the Flow

This file is the exact bridge from mathematics to implementation.

### Why It Is Important

If you want to understand the algorithm, this is the most important file.

This is where:

- the formula is translated into code
- the joint coupling actually appears
- the optimization variables interact mathematically

### Key Questions To Ask While Reading

- where is the reconstruction term coded?
- where is the classifier regularization coded?
- where is the coupling through `q_j` coded?
- how does the gradient of `C` differ from a pure dictionary-learning model?

---

## 7.5 `src/prox.py`

### Purpose

Implement the proximal operators for the nonsmooth terms.

### Main Responsibilities

- soft-threshold `C`
- clip `D`
- apply the nonnegative shifted shrinkage for `u`

### Main Functions

- `prox_C`
- `prox_D`
- `prox_u`

### In the Flow

This file is used after the gradient step in every solver iteration.

### What Each Prox Means

- `prox_C`: impose sparsity
- `prox_D`: enforce `D in [0,1]`
- `prox_u`: enforce hinge-like penalty and nonnegativity

---

## 7.6 `src/solver.py`

### Purpose

Run joint proximal gradient with backtracking.

### Main Responsibilities

- compute gradients
- take a gradient step
- apply proximal operators
- perform backtracking line search
- record objective history
- stop on convergence or iteration limit

### Main Functions

- `gradient_step`
- `prox_step`
- `fit_joint_pg`

### In the Flow

This file is the optimization engine of the joint method.

### Conceptual Iteration

1. evaluate current objective
2. compute gradients of the smooth part
3. try a step size
4. take a gradient step
5. apply prox operators
6. check sufficient decrease
7. accept or shrink step
8. record history

### What To Watch Closely

- how the smooth objective is used in the backtracking condition
- how each variable block is updated
- why `C, D, u` get prox operators but `w, b` do not

---

## 7.7 `src/baselines.py`

### Purpose

Implement the non-joint comparison methods.

### Main Responsibilities

- raw linear SVM baseline
- separate dictionary-learning baseline
- code inference for test samples with a fixed dictionary

### Main Functions

- `fit_raw_svm`
- `fit_separate_dictionary`
- `fit_separate_dict_svm`

### In the Flow

This file gives the comparison methods needed to evaluate whether the joint method is actually useful.

### Why It Matters

This file is where you can see the exact difference between:

- using representation learning without classifier coupling
- using no representation learning at all
- using the fully joint method

### Key Difference From Joint

In the separate method:

- dictionary learning ignores classifier information
- classifier training happens only after codes are already fixed

---

## 7.8 `src/metrics.py`

### Purpose

Compute evaluation metrics and support analysis.

### Main Responsibilities

- predict from codes
- compute classification accuracy
- compute reconstruction error
- compute code sparsity
- evaluate a trained joint model on a new split by inferring codes

### Main Functions

- `predict_from_codes`
- `accuracy_from_codes`
- `reconstruction_error`
- `code_sparsity`
- `infer_codes_with_dictionary`
- `evaluate_joint_model`
- `summarize_joint_result`

### In the Flow

This file is used after training to evaluate both train and test performance.

### Important Subtlety

For the joint model, test accuracy is not obtained from training codes.

Instead:

1. keep learned `D, w, b`
2. infer test codes with fixed `D`
3. classify using `w, b`

This makes the comparison fairer.

---

# 8. Notebook Roles

## `notebooks/01_problem_setup.ipynb`

Purpose:

- define the task
- verify data loading
- check shapes and configuration

## `notebooks/02_baselines.ipynb`

Purpose:

- run raw SVM
- run separate dictionary + SVM

## `notebooks/03_joint_optimization.ipynb`

Purpose:

- run the joint optimization algorithm
- inspect convergence and immediate metrics

## `notebooks/04_results_analysis.ipynb`

Purpose:

- compare all methods with a unified metric view
- inspect train/test accuracy
- inspect convergence curves
- visualize learned dictionary atoms

---

# 9. Method Comparison Summary

| Method | Learns Dictionary? | Learns Codes? | Uses Classifier During Representation Learning? |
|---|---|---|---|
| Raw SVM | No | No | No |
| Separate Dict + SVM | Yes | Yes | No |
| Joint Dict + SVM | Yes | Yes | Yes |

This table is one of the most important conceptual summaries in the whole project.

---

# 10. How To Read The Repo Efficiently

Recommended order:

1. `plan.md`
2. this note
3. `src/data.py`
4. `src/init.py`
5. `src/model.py`
6. `src/prox.py`
7. `src/solver.py`
8. `src/metrics.py`
9. `src/baselines.py`
10. `notebooks/*`
11. `tests/*`

Do not start by reading line by line blindly.

For each file, answer:

- what is this file responsible for?
- where does it sit in the flow?
- what are its input and output shapes?
- which formula does it implement?

---

# 11. Suggested Notion Structure

If you import this into Notion, use this page as the top-level overview.

Then create child pages:

- `Repo Flow Overview`
- `Math to Code Mapping`
- `Method Comparison`
- `src/data.py`
- `src/init.py`
- `src/model.py`
- `src/prox.py`
- `src/solver.py`
- `src/baselines.py`
- `src/metrics.py`

For each file page, use this template:

## Purpose

What does this file do?

## Position in Flow

Where does this file sit in the overall pipeline?

## Main Functions

| Function | Input | Output | Role |
|---|---|---|---|

## Important Shapes

- ...

## Math Correspondence

- ...

## Questions / Confusions

- ...

---

# 12. Questions Worth Recording While Studying

These are good questions to keep in your own notes:

- Why is the joint method optimized with proximal gradient instead of a standard solver?
- Why is only `C` sparse and not `D`?
- Why does `u` need a proximal operator?
- Why is the backtracking condition checked on the smooth part?
- How different is the learned representation between separate and joint methods?
- Why might raw SVM still outperform the current joint model on MNIST?

---

# 13. Final One-Sentence Summary

This repo implements a binary image-classification pipeline in which dictionary learning, sparse coding, and linear classification are either trained separately or coupled inside a single proximal-gradient-based joint optimization framework.
