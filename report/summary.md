The provided sources discuss the challenges of applying traditional batch Robust Principal Component Analysis (RPCA) to large, streaming datasets and detail the evolution of online RPCA algorithms designed to handle different types of time-varying data structures.

### Summary of RPCA Algorithms

**Robust Principal Component Analysis (RPCA) and Algorithm 1 (ALM)**
RPCA, particularly the Principal Component Pursuit (PCP) formulation, decomposes an observed data matrix $M$ into a **low-rank component $L$** (stable background/subspace) and a **sparse component $S$** (outliers/noise/moving objects) by minimizing $\|L\|_* + \lambda\|S\|_1$ subject to $L + S = M$.

**Algorithm 1: Principal Component Pursuit by Augmented Lagrangian Multiplier (ALM)** is a standard **batch algorithm** used to solve the convex PCP formulation. It achieves high accuracy but is computationally expensive, requires loading all observed data into memory, and does not scale well for big data. The updates involve closed-form solutions: Singular Value Thresholding ($D_{\tau}$) for the nuclear norm and Soft-Thresholding ($S_{\tau}$) for the $\ell_1$ norm.

**Algorithms 2 & 3: RPCA-STOC (Online Robust PCA via Stochastic Optimization)**
RPCA-STOC is an **online algorithm** designed to overcome the memory and speed limitations of batch methods. It operates by processing one sample per time instance and using stochastic optimization techniques, often minimizing a relaxed objective function where the low-rank matrix $L$ is factored as $UV$.
*   **Limitation:** RPCA-STOC explicitly assumes a **stable subspace**. Since it equally weights all previously observed samples in its loss minimization, the result is an "average" subspace that struggles to adapt if the underlying structure changes over time, causing its performance to deteriorate.

**Algorithm 4: Online Moving Window RPCA (OMWRPCA)**
OMWRPCA was developed to address the limitations of RPCA-STOC regarding changing subspaces. It is an online method that updates the subspace basis $U$ by minimizing an empirical loss based only on a user-specified **moving window of the most recent $n_{win}$ samples**. This design allows OMWRPCA to successfully track **slowly changing subspaces**.

**Algorithm 5: Online Moving Window RPCA with Change Point Detection (OMWRPCA-CP)**
OMWRPCA-CP extends the moving window concept to handle **abruptly changed subspaces**, a weakness of the basic OMWRPCA.
*   **Mechanism:** It embeds **hypothesis testing** by monitoring the support size (number of nonzero elements, $c_t$) of the sparse vector $S_t$. When an abrupt subspace change occurs, the current subspace model fails, resulting in an abnormally large sparse vector support size.
*   **Action:** When a change point is detected (i.e., $c_t$ is abnormally high for a consecutive period), the algorithm **restarts the OMWRPCA process** from that detected time point, allowing it to re-estimate the rank and basis for the new underlying subspace.
*   OMWRPCA-CP is the first algorithm known to simultaneously detect change points and handle both slowly changing and abruptly changed subspaces in an online fashion.

***

### Comparison Table of RPCA Algorithms

| Algorithm | Idea/Mechanism | Pros | Cons | When to Use (Background Requirement) | Moving Object/Sparse Component Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alg. 1: ALM (Batch)** | Solves $\min \|L\|_* + \lambda\|S\|_1$ subject to $L+S=M$ using iterative thresholding (SVT, Soft-Thresholding). | High accuracy; theoretically rigorous. | Requires all data in memory; slow/inefficient for big data; scales poorly. | **Static or Known Structure** (Offline analysis of a fixed dataset). | Recovers the entire sparse matrix $S$ for the observed time window. |
| **Alg. 2/3: RPCA-STOC** | Online update using stochastic optimization based on minimizing empirical loss over *all* historical samples. | Fast computation $O(mr^2)$; low memory cost $O(mr)$. | **Requires a strictly stable/static subspace**; performance degrades if subspace changes. | **Stable/Static Subspace Only** (e.g., synthetic data or unchanging scenes). | Updates the sparse vector $s_t$ for each new sample. |
| **Alg. 4: OMWRPCA** | Online update minimizing empirical loss based only on a fixed **moving window** ($n_{win}$) of the most recent samples. | Efficiently tracks **slowly changing subspaces** (e.g., gradual illumination changes). | Fails if the subspace changes abruptly (e.g., scene change or sudden rank change). | **Slowly Changing Subspaces**. | Updates the sparse vector $s_t$ for each new sample. |
| **Alg. 5: OMWRPCA-CP** | OMWRPCA augmented with **hypothesis testing** on the sparse component's support size ($c_t$) to detect change points and initiate subspace restarts. | **Handles both slowly and abruptly changed subspaces**; accurately identifies change points. | Requires tuning multiple specialized parameters ($N_{check}, \alpha_{prop}$, etc.). | **Dynamic Subspaces** (including abrupt scene changes or system failures). | Monitoring the **support size $c_t$** of the sparse vector $s_t$ is key to detecting changes. |