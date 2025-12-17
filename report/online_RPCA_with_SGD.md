+ assumed the background stable, subspace U is stable. so this can run online with assumption U is fixed

# TODO: recheck this one
The online algorithm for Robust PCA via Stochastic Optimization (RPCA-STOC) proposed by Feng et al. addresses the original batch problem by solving a reformulated optimization problem sequentially for each incoming data sample.

### 1) Why Separation of $S, v$ and $U$ is Possible

The separation into sequential minimization steps—first finding the foreground and coefficients $(s, v)$, and then updating the basis $U$—is a core feature of the **stochastic optimization** approach used in RPCA-STOC. This approach requires reformulating the original low-rank matrix $L$ using a basis $U$ and coefficients $V$: $L = UV$.

The general optimization objective involves minimizing the empirical loss over all historical samples $M_t = [m_1, \dots, m_t]$ by splitting the variables:
$$\min_{U, V, S} \frac{1}{2} \|M - UV - S\|_F^2 + \frac{\lambda_1}{2} (\|U\|_F^2 + \|V\|_F^2) + \lambda_2 \|S\|_1$$

In the online setting, this problem is solved iteratively, treating the processing of each new sample $m_t$ as a step in minimizing the empirical loss. The overall process is structured into two alternating blocks:

1.  **Projecting the Sample (Finding $v_t$ and $s_t$):** At time $t$, the algorithm reveals a new sample $m_t$. Assuming the current basis $U^{t-1}$ is fixed, the algorithm finds the sparse error vector $s_t$ and the coefficient vector $v_t$ by minimizing the loss specific to that single sample:
    $$(v_t, s_t) \leftarrow \arg\min_{v, s} \frac{1}{2} \|m_t - U^{t-1}v - s\|_2^2 + \frac{\lambda_1}{2} \|v\|_2^2 + \lambda_2 \|s\|_1$$
    This step effectively determines how the new observation $m_t$ is decomposed into a sparse foreground component $s_t$ and a low-rank background component $l_t = U^{t-1}v_t$.

2.  **Updating the Basis (Finding $U^t$):** Once all necessary sparse components $s_i$ and coefficients $v_i$ from $i=1$ to $t$ are known, the optimization proceeds to update the basis $U^t$ by minimizing an accumulating function $g_t(U)$. This step relies on the pre-computed pairs $\{v_i, s_i\}_{i=1}^t$ to update the representation of the low-rank subspace.

This block-coordinate descent approach allows the optimization to be performed sequentially: first, processing the current data vector using the current basis estimate, and second, updating the basis using the accumulated data and decomposition results.

### 2) Derivation of the Closed-Form Formula for $U^t$

The explicit solution for the basis $U^t$ is obtained by treating the coefficient vectors $\{v_i\}$ and sparse vectors $\{s_i\}$ as fixed constants and performing **linear least squares with $\ell_2$ regularization** (ridge regression) on the minimization function $g_t(U)$.

The function minimized to find $U^t$ is $g_t(U)$, which depends on the historical data and estimates:
$$g_t(U) = \frac{1}{t} \sum_{i=1}^t \left( \frac{1}{2} \|m_i - Uv_i\|_2^2 + \frac{\lambda_1}{2} \|v_i\|_2^2 + \lambda_2 \|s_i\|_1 \right) + \frac{\lambda_1}{2t} \|U\|_F^2$$

Since terms not involving $U$ are irrelevant to the minimization (namely, terms involving $m_i$, $v_i$, $s_i$, or constants), the core minimization problem is equivalent to finding the $U$ that solves:
$$\min_U \sum_{i=1}^t \frac{1}{2} \|(m_i - s_i) - Uv_i\|_2^2 + \frac{\lambda_1}{2} \|U\|_F^2$$
Let $y_i = m_i - s_i$ represent the corrected low-rank data vector ($l_i$) derived from the total observation $m_i$ and sparse error $s_i$. The objective becomes:
$$\min_U \sum_{i=1}^t \frac{1}{2} \|y_i - Uv_i\|_2^2 + \frac{\lambda_1}{2} \|U\|_F^2$$

**Step 1: Calculate the Gradient**
To find the optimal $U^t$, we calculate the gradient of this objective with respect to $U$ and set it to zero. Using matrix calculus rules for the Frobenius norm squared and the $\ell_2$ regularization term:
$$\nabla_U \left[ \sum_{i=1}^t \frac{1}{2} \|y_i - Uv_i\|_2^2 + \frac{\lambda_1}{2} \|U\|_F^2 \right] = 0$$
$$\sum_{i=1}^t \left[ U v_i v_i^T - y_i v_i^T \right] + \lambda_1 U = 0$$

**Step 2: Solve the Linear System**
Rearranging the terms, we isolate $U$:
$$U \left( \sum_{i=1}^t v_i v_i^T \right) + \lambda_1 U = \sum_{i=1}^t y_i v_i^T$$
Factor $U$ on the left side:
$$U \left( \sum_{i=1}^t v_i v_i^T + \lambda_1 I \right) = \sum_{i=1}^t (m_i - s_i) v_i^T$$
(Note that $I$ is the identity matrix of the appropriate size, where $\lambda_1 I$ arises from the $\lambda_1 \|U\|_F^2$ regularization term.)

**Step 3: Obtain the Closed Form**
Assuming the matrix $\left( \sum_{i=1}^t v_i v_i^T + \lambda_1 I \right)$ is invertible (which is guaranteed by the positive penalty parameter $\lambda_1$), the solution is obtained by post-multiplying both sides by the inverse:
$$U^t = \left[ \sum_{i=1}^t (m_i - s_i) v_i^T \right] \left[ \left(\sum_{i=1}^t v_i v_i^T \right) + \lambda_1 I \right]^{-1}$$

This derived solution matches the explicit formula provided in your query. This minimization corresponds to the step described in Algorithm 2, Line 7, where $U^t$ minimizes the quadratic function $\frac{1}{2} \operatorname{Tr}[U^T(A_t + \lambda_1 I)U] - \operatorname{Tr}(U^T B_t)$, which is the matrix form of the expression derived above.