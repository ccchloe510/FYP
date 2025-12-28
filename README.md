# 1. Joint Optimisation Model

**Objective Function:**
$$
\min_{D, A, w, b} \frac{1}{2}\|X - DA\|_F^2 + \lambda \|A\|_1 + C \sum_{i=1}^N \max(0, 1 - y_i(w^\top a_i + b)) + \frac{\gamma}{2} \|w\|_2^2
$$

* **Step A (Dictionary):** Learn shared atoms to represent features (e.g., MNIST numbers).
* **Step B (Sparse Code):** Each picture is a sparse vector (which element used and how much).
* **Step C (SVM):** Input is sparse code, output is classification.

**Definitions:**
* $X = [x_1, \dots, x_n] \in \mathbb{R}^{d \times n}$ (Original data, $d$ dimensions, $n$ samples).
* $D = [d_1, \dots, d_k] \in \mathbb{R}^{d \times k}$ (Dictionary, $k$ atoms).
    * Constraint: $\|d_j\|_2 \le 1$ to prevent overscaling (large $D$, small $A$).
* $A = [a_1, \dots, a_n] \in \mathbb{R}^{k \times n}$ (Sparse representations).
    * $\lambda \uparrow \Rightarrow$ more sparse.
    * $\lambda \downarrow \Rightarrow$ more dense.
* $C$: Soft-margin weight.
    * Large $C$: Care about classification accuracy (pure SVM).
    * Small $C$: Care about generalization (degrades to dictionary learning).

---

# 2. A-update (ADMM)

Fix $D, w, b$. The objective is:
$$
\min_{A} \frac{1}{2} \|X - DA\|_F^2 + \lambda \|A\|_1 + \mu C \sum_{i=1}^n \max(0, 1 - y_i(w^\top a_i + b))
$$

Separate into columns (independent per sample $i$):
$$
\min_{a_i} \frac{1}{2} \|x_i - D a_i\|_2^2 + \lambda \|a_i\|_1 + \mu C \max(0, 1 - y_i(w^\top a_i + b))
$$

### 2.1 Reformulation
Introduce auxiliary variables to handle non-smooth terms:
* Let $p_i = a_i$ (for $\ell_1$ norm).
* Let $z_i = 1 - y_i(w^\top a_i + b)$ (for hinge loss).

**Problem becomes:**
$$
\min_{a, p, z} \frac{1}{2} \|x - Da\|_2^2 + \lambda \|p\|_1 + \mu C \max(0, z)
$$
**Subject to:**
1.  $a - p = 0$
2.  $y(w^\top a + b) + z - 1 = 0$

### 2.2 Augmented Lagrangian (Scaled Form)
Introduce dual variables $u$ (for $a-p$) and $v$ (for classification constraint), and penalties $\rho_1, \rho_2 > 0$:

$$
\mathcal{L}(a, p, z; u, v) = \frac{1}{2} \|x - Da\|_2^2 + \lambda \|p\|_1 + \mu C \max(0, z) + \frac{\rho_1}{2} \|a - p + u\|_2^2 + \frac{\rho_2}{2} (y(w^\top a + b) + z - 1 + v)^2
$$

---

## 3. ADMM Iteration Steps

### Step 1: Update $a$ (Quadratic Subproblem)
Solve for $a^{t+1}$ by minimizing terms involving $a$.
Terms:
1.  $\frac{1}{2} \|x - Da\|_2^2 \Rightarrow \text{derivative: } D^\top(Da - x)$
2.  $\frac{\rho_1}{2} \|a - p^t + u^t\|_2^2 \Rightarrow \text{derivative: } \rho_1(a - p^t + u^t)$
3.  Let $r(a) = y(w^\top a + b) + z^t - 1 + v^t$. Note $\nabla_a r(a) = yw$.
    Derivative of $\frac{\rho_2}{2} r(a)^2$ is:
    $$
    \rho_2 r(a) (yw) = \rho_2 (w^\top a + b + y(z^t - 1 + v^t)) w \quad (\text{using } y^2=1)
    $$

Set sum of derivatives to 0:
$$
D^\top(Da - x) + \rho_1(a - p^t + u^t) + \rho_2 w (w^\top a + b + y(z^t - 1 + v^t)) = 0
$$

**Linear System Solution:**
$$
(D^\top D + \rho_1 I + \rho_2 ww^\top)a^{t+1} = D^\top x + \rho_1(p^t - u^t) - \rho_2(b + y(z^t - 1 + v^t))w
$$
*(Use CG in code)*

---

### Step 2: Update $p$ (Proximal of $\ell_1$)
$$
p^{t+1} = \arg\min_p \lambda \|p\|_1 + \frac{\rho_1}{2} \|p - (a^{t+1} + u^t)\|_2^2
$$
Let $q = a^{t+1} + u^t$. The objective is $f(p) = \lambda |p| + \frac{\rho_1}{2}(p - q)^2$.

**Derivation:**
1.  **If $p > 0$**: $f = \lambda p + \frac{\rho_1}{2}(p - q)^2$
    $$
    \frac{df}{dp} = \lambda + \rho_1(p - q) = 0 \Rightarrow p = q - \frac{\lambda}{\rho_1}
    $$
    Valid only if $q > \frac{\lambda}{\rho_1}$.
2.  **If $p < 0$**: $f = -\lambda p + \frac{\rho_1}{2}(p - q)^2$
    $$
    \frac{df}{dp} = -\lambda + \rho_1(p - q) = 0 \Rightarrow p = q + \frac{\lambda}{\rho_1}
    $$
    Valid only if $q < -\frac{\lambda}{\rho_1}$.
3.  **If $p = 0$**: Optimal when $-\frac{\lambda}{\rho_1} \le q \le \frac{\lambda}{\rho_1}$.

**Result (Soft-thresholding):**
$$
p^{t+1} = \mathcal{S}_{\lambda/\rho_1}(q) = \text{sign}(q)\max(|q| - \frac{\lambda}{\rho_1}, 0)
$$

---

### Step 3: Update $z$ (Proximal of Hinge Loss)
$$
z^{t+1} = \arg\min_z \mu C \max(0, z) + \frac{\rho_2}{2} (y(w^\top a^{t+1} + b) + z - 1 + v^t)^2
$$
Let $q = 1 - y(w^\top a^{t+1} + b) - v^t$.
Simplified problem:
$$
z^{t+1} = \arg\min_z \mu C \max(0, z) + \frac{\rho_2}{2} (z - q)^2
$$

**Derivation:**
1.  **Case $z < 0$**: $\max(0, z) = 0$.
    $$
    \min \frac{\rho_2}{2}(z - q)^2 \Rightarrow z = q
    $$
    Valid if $q < 0$.
2.  **Case $z > 0$**: $\max(0, z) = z$.
    $$
    \min \mu C z + \frac{\rho_2}{2}(z - q)^2 \Rightarrow \frac{d}{dz} = \mu C + \rho_2(z - q) = 0 \Rightarrow z = q - \frac{\mu C}{\rho_2}
    $$
    Valid if $q > \frac{\mu C}{\rho_2}$.
3.  **Case $z = 0$**: Optimal when $0 \le q \le \frac{\mu C}{\rho_2}$.

**Result:**
$$
z^{t+1} = \begin{cases} 
q & q < 0 \\
0 & 0 \le q \le \frac{\mu C}{\rho_2} \\
q - \frac{\mu C}{\rho_2} & q > \frac{\mu C}{\rho_2}
\end{cases}
$$

---

### Step 4: Dual Updates
Update dual variables based on residuals:
1.  $u^{t+1} = u^t + (a^{t+1} - p^{t+1})$
2.  $v^{t+1} = v^t + (y(w^\top a^{t+1} + b) + z^{t+1} - 1)$

---

# 4. D-update (Dictionary)

Objective:
$$
\min_D f(D) = \frac{1}{2} \|X - DA\|_F^2
$$
Using trace properties $\frac{1}{2}\|E\|_F^2 = \frac{1}{2}\text{Tr}(E^\top E)$:
$$
f(D) = \frac{1}{2} \text{Tr}((X - DA)^\top (X - DA))
$$
$$
= \frac{1}{2} [\text{Tr}(X^\top X) - \text{Tr}(X^\top DA) - \text{Tr}(A^\top D^\top X) + \text{Tr}(A^\top D^\top DA)]
$$
Gradient $\nabla_D f(D)$:
$$
-XA^\top + DAA^\top = 0 \Rightarrow DAA^\top = XA^\top
$$

**Solution:**
1.  If $AA^\top$ is invertible: $D = XA^\top(AA^\top)^{-1}$
2.  If non-invertible, use **Ridge Regression**:
    $$
    D = XA^\top(AA^\top + \epsilon I)^{-1}
    $$
3.  **Column Normalization:**
    $$
    d_j \leftarrow \frac{d_j}{\max(\|d_j\|_2, 1)}
    $$

---

# 5. Classifier Update (w, b)

We use **Squared Hinge Loss** for differentiability (smoothing).
Let $\tilde{C} = \mu C$.
$$
\min_{w, b} F(w, b) = \frac{1}{2} \|w\|_2^2 + \tilde{C} \sum_{i=1}^n \max(0, 1 - y_i(w^\top a_i + b))^2
$$

**Definitions:**
* Decision: $g_i(w, b) = w^\top a_i + b$
* Margin violation: $s_i(w, b) = 1 - y_i g_i(w, b)$
* Loss: $L_i(w, b) = \max(0, s_i)^2$
* Active set: $I(w, b) = \{i : s_i(w, b) > 0\}$

**Gradients:**
1.  **w.r.t $w$**:
    $$
    \nabla_w L_i = 2 s_i \nabla_w s_i = 2 s_i (-y_i a_i)
    $$
    $$
    \nabla_w F = w + 2\tilde{C} \sum_{i \in I} s_i (-y_i a_i)
    $$
2.  **w.r.t $b$**:
    $$
    \nabla_b L_i = 2 s_i \nabla_b s_i = 2 s_i (-y_i)
    $$
    $$
    \frac{\partial F}{\partial b} = 2\tilde{C} \sum_{i \in I} s_i (-y_i)
    $$

---

### Optimization: L-BFGS

Let $\theta = [w^\top, b]^\top$. Newton's method ($\theta_{t+1} = \theta_t - H^{-1} \nabla F$) is too expensive.
Use **Quasi-Newton (L-BFGS)**:
1.  $s_t = \theta_{t+1} - \theta_t$
2.  $y_t = \nabla F(\theta_{t+1}) - \nabla F(\theta_t)$
3.  Secant condition: $B_{t+1} s_t = y_t$
4.  Store only nearest $m$ pairs $(s, y)$.

**Two-Loop Recursion (to find $p_t = -H_t \nabla F(\theta_t)$):**
1.  Initialize $q = \nabla F(\theta_t)$
2.  **Backward loop** ($i = t \dots t-m$):
    * $\alpha_i = \frac{s_i^\top q}{y_i^\top s_i}$
    * $q \leftarrow q - \alpha_i y_i$
3.  Set $H_0 = \gamma_t I$ where $\gamma_t = \frac{s_{t-1}^\top y_{t-1}}{y_{t-1}^\top y_{t-1}}$
4.  $r = H_0 q$
5.  **Forward loop**:
    * $\beta_i = \frac{y_i^\top r}{y_i^\top s_i}$
    * $r \leftarrow r + s_i(\alpha_i - \beta_i)$
6.  Result direction $p_t = -r$

**Line Search:**
Update $\theta_{t+1} = \theta_t + \lambda_t p_t$ using **Armijo backtracking**.