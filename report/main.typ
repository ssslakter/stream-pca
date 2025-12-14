#import "@preview/obsidius:0.1.0": *

#show: notes.with("Report");

#align(center)[#heading(numbering: none)[Optimization Project: Compressive Robust PCA]]

#v(2em)
= Problem Statement

The goal of this project is to perform *Background Subtraction* on video streams. Mathematically, a video can be represented as a matrix $M$, where each column is a flattened video frame. We model this matrix as the sum of two distinct components:
$ M = L + S $
where:
- $L$ is a *Low-Rank* matrix representing the static background (correlated across frames).
- $S$ is a *Sparse* matrix representing the foreground objects (moving people, cars, noise).

To recover these components, we solve the *Robust Principal Component Analysis (RPCA)* problem:

$ min_(L, S) norm(L)_* + lambda norm(S)_1 quad "s.t." quad L + S = M $

where $norm(dot)_*$ is the Nuclear Norm (sum of singular values) encouraging low rank, and $norm(dot)_1$ is the $L_1$ norm encouraging sparsity.

= Optimization Framework: ADMM

Since the objective function contains non-smooth terms ($L_1$ and Nuclear norms), Gradient Descent is not applicable. We utilize the *Alternating Direction Method of Multipliers (ADMM)*.

We form the Augmented Lagrangian:
$ 
cal(L)(L, S, Y) = norm(L)_* + lambda norm(S)_1 + chevron.l Y, M - L - S chevron.r + rho/2 norm(M - L - S)_F^2 
$
where $Y$ is the dual variable and $rho$ is the penalty parameter. ADMM solves this by iterating three steps:

1. *L-Update (Background):* Minimizing for $L$ leads to the *Singular Value Thresholding (SVT)* operator:
   $ L^(k+1) = cal(D)_(1/rho) (M - S^k + 1/rho Y^k) $
   
2. *S-Update (Foreground):* Minimizing for $S$ leads to the *Soft Thresholding* operator:
   $ S^(k+1) = cal(S)_(lambda/rho) (M - L^(k+1) + 1/rho Y^k) $

3. *Dual Update:*
   $ Y^(k+1) = Y^k + rho (M - L^(k+1) - S^(k+1)) $

= Compressive Online Modification

While standard ADMM guarantees the global optimum, it requires storing the entire video in memory (Batch Mode) and performing expensive SVDs. To achieve real-time performance on high-resolution video, we implement a *Compressive Online* approach.

== Basis Factorization
We assume the low-rank background $L$ lies in a subspace spanned by a basis $U$:
$ L = U v $
Instead of learning $U$ iteratively via SVD, we use a *Robust Median Initialization*. Since the background is static for the majority of the time, the pixel-wise median of $N$ frames provides a mathematically robust estimate of the subspace $U$.

== Compressive Sensing Projection
For each incoming frame $x$ (vectorized as $m$), we want to find the scaling coefficient $v$ (representing lighting changes) that best fits the background basis $U$. 

To accelerate computation, we apply a *Compressive Mask*. We only observe a subset of pixels $Omega$ (where $|Omega| approx 0.1 dot d$). We solve the projection problem on this subset:

$ v = arg min_v norm(P_Omega (m) - v dot P_Omega (u))_2^2 $

This has a closed-form solution:
$ v = (chevron.l u_Omega, m_Omega chevron.r) / (chevron.l u_Omega, u_Omega chevron.r + epsilon) $

== Sparse Reconstruction
Once $v$ is computed, we reconstruct the full background $L = v dot U$ (recovering pixels we did not even sample). The foreground is extracted using a thresholding operator similar to the ADMM S-step, implemented via a shifted ReLU to act as a "Digital Gate":

$ S = "ReLU"(|M - L| - tau) $
