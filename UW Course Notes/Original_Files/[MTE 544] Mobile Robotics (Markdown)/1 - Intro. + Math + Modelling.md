# 1. Intro. + Math + Modelling

[TOC]

---

## Linear Systems

### Linear Algebra Table

|                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Scalar**<br />$c \in \mathbb{R}$                           | **Vector**<br />$b = \begin{bmatrix} b_1\\b_2\\\vdots\\b_n\end{bmatrix} \in \mathbb{R}^n$ | **<br />Matrix**<br />$A = \begin{bmatrix} A_{11} & A_{12} & \dots \\ A_{21} & \ddots\\ \vdots & & A_{nm}\end{bmatrix} \in \mathbb{R}^{n\times m}$<br /><br />- Fat matrix: n<m<br />- Skinny matrix: n>m |
| <br />**Matrix Transpose**<br />$A^T = \begin{bmatrix} A_{11} & A_{21} & \dots \\ A_{12} & \ddots\\ \vdots & & A_{mn} \end{bmatrix}$ | <br />**Matrix Addition**<br />$A+B = \begin{bmatrix} A_{11} + B_{11} & A_{12} + B_{12} & \dots \\ A_{21} + B_{21} & \ddots\\ \vdots & & A_{mn} + B_{mn} \end{bmatrix}$ | <br />**Matrix Multiplication**<br />$AB = \begin{bmatrix} \sum_i A_{1i}B_{i1} & \sum_iA_{1i}B_{i2} & \dots \\ \sum_i A_{2i}B_{i1} & \ddots\\ \vdots & & \sum_i A_{mi}B_{in} \end{bmatrix}$ |
| **Matrix transpose of Added Matrices**<br />$(A+ B)^T = A^T + B^T$ | **Matrix Transpose of Multiplied Matrices**<br />$(AB)^T = B^T A^T$ | <br />**Quadratic Form**<br />$\begin{align}(Ax + b)^T (Ax + b)&=x^TA^TAx + 2x^T A^T v + b^Tb \\ &= \underbrace{x^TCx}_{\text{quadratic term}}+ d^T x + e\end{align}$ |
| <br />**Matrix Rank:** $\rho(A)$<br /><br />- The number of independent rows or columns<br />- **Nonsingular** = Full Rank : $\rho(A)=\min(n,m)$<br />- **Singular** = Not Full Rank: $\rho(A)<\min(n,m)$<br />-- Non-empty nullspace: $\exist x \text{ such that } Ax=0$ | **Matrix Inverse (square A)**<br />$AA^{-1} = A^{-1}A = I$<br /><br />- **Nonsingular (Full Rank) and square<br />.  => INVERTIBLE** | **Symmetric Matrix**<br /><br />$A = A^T = \begin{bmatrix} A_{11} & A_{12} & \dots \\ A_{21} & \ddots\\ \vdots & & A_{nm}\end{bmatrix}$ |
| <br />**Matrix Trace**<br /><br />$tr(A) = \sum_i A_{ii}$    | **Positive Definiteness (Semi-Definiteness)**<br />- For a symmetric $n\times n$ matrix $A$, and for any $x$ in $\mathbb{R}^n$:<br />$x^T Ax > 0 \qquad (x^T Ax \geq 0)$ | <br />**Eigenvalues and Eigenvectors** of a matrix<br />- For a matrix $A$, <br />--- the vector $x$ is an *eigenvector* of $A$ <br />--- with a corresponding *eigenvalue* $\lambda$  <br />-- if they satisify the equation: $Ax = \lambda x$<br /><br />NOTE:<br />- The eigenvalues of a diagonal matrix are its diagonal elements<br />- The inverse of $A$ exists iff none of the eigenvalues are zero<br />--- ($\lambda_i \neq 0$)<br />- Positive definite $A$ has all eigenvalues greater than zero <br />--- ($\lambda_i > 0 \, \forall i$) |
| <br />**Differentiation of Linear Matrix Equation**<br />$\frac{d}{dx} (Ax) = A$<br />$\frac{d}{dx} (A^Tx) = A^T$<br /> | **Differentiation of a quadratic matrix equation**<br />$\frac{d}{dx}(x^T Ax) = x^TA + x^T A^T$ |                                                              |

#### Least Squares Solution

- If $A$ is a skinny matrix (n>m), and we wish to find $x$ for which $Ax = b$

- Since $A$ is skinny, the problem is **over-constrained** => ==**No solution exists**==

- Instead, minimize the square of the error between $Ax$ and $b$:

  - $$
    \begin{align}
    \min_x & {||Ax - b||^2_2} \\
    &= \min_x (Ax-b)^T (Ax-b)\\
    &= \min_x x^T A^T Ax - 2b^T Ax + b^Tb
    \end{align}
    $$

- Setting the derivative to zero

  - $$
    \begin{align}
    2x^T A^T A - 2b^T A &= 0\\
    A^T Ax &= A^T b\\
    x&=(A^TA)^{-1}A^Tb\\
    x=A^{\dagger}b
    \end{align}
    $$

  - $A^{\dagger}$  : **Pseudo-inverse**

- This terminology is used over and over, Quadratic cost minimized to find closed form solution





---

## Probability

### Probability Algebra Table

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Measures of Distributions** $\mathcal{N}(\mu, \sigma^2)$<br /><br />**Mean:**<br />- Expected value of a random variable: <br />                $\mu = E[x]$ <br />-- Discrete Case: $\mu = \sum_{i=1}^n x_i p(x_i)$<br />-- Continuous Case: $\mu = \int xp(x) dx$<br /><br />**Variance:**<br />- Measure of the variability of a random variable: <br />                $\sigma^2(x) = E[(x-\mu)^2]$<br />-- Discrete Case: $\sigma^2(x) = \sum^n_{i=1} (x_i - \mu)^2 p(x_i)$<br />-- Continuous Case: $\sigma^2(x) = \int (x - \mu)^2 p(x) dx$ | **<br />Multi-variable Distributions**<br /><br />**Mean:**<br />$\mu = \begin{bmatrix}\mu_1 \\ \vdots \\ \mu_n \end{bmatrix} = \begin{bmatrix}E[X_1] \\ \vdots \\ E[X_n]\end{bmatrix}$<br /><br />**Covariance:**<br />- Measure of how much two random variables change together:<br />.     $Cov(X_i, X_j) = E[(X_i-\mu_i)(X_j - \mu_j)] = E[X_i X_j] - \mu_i \mu_j$<br />-- If $Cov(X,Y)>0$, when $X$ is above its expected values, then $Y$ tends to be above its expected value<br />-- If $Cov(X,Y) < 0$, when $X$ is above its expected value, then $Y$ tends to be below its expected value<br />-- If $X,Y$ are independent, <br /><br />**Covariance Matrix, $\Sigma$**<br />- Defins variational relationship between each pair of random variables: $\Sigma_{i,j} = Cov(X_i, X_j)$<br />- Generalization of variance, diagonal elements represent variance of each random variable: $Cov(X_i, X_i) = Var(X_i)$<br />- Covariance matrix is **Symmetric, Positive semi-definite** |
| <br />**Multiplication by a constant matrix yeilds**<br />$\begin{align}cov(Ax) &= E[(Ax - A\mu)(Ax - A\mu)^T] \\&= E[A(x-\mu)(x-\mu)^T A^T]\\&= AE[(x-\mu)(x-\mu)^T]A^T \\&= A \,cov(x)\, A^T\end{align}$ | <br />**Addition/ Subtraction of random variables**<br />$cov(X\pm Y) = cov(X) + cov(Y) \pm 2 cov(X,Y)$<br /><br />- If X,Y independent,<br />$cov(X\pm Y) = cov(X) + cov(Y)$ |
| **Joint Probability**<br />- Probability of x and y:<br />$p( X = x \text{ and } Y = y ) = p( x, y )$ | **Independence**<br />- If X,Y are independent, then<br />$p ( x, y ) = p ( x ) p ( y )$ |
| **Conditional Probability**<br />- Probability of x given y:<br />$p(X=x | Y=y)=p(x|y)$<br /><br />- Relation to joint probability:<br />$p(x|y) = \frac{p(x,y)}{p(y)}$<br />- If X and Y are independent,<br />$p(x|y) = p(x)$ | <br />**Law of Total Probability**<br /><br />- **==Discrete==**:<br />$\underset{x}{\sum} p(x) = 1$<br />$p(x) = \underset{y}{\sum} p(x,y)$<br />$p(x) = \underset{y}{\sum} p(x|y) p(y)$<br /><br />- ==**Continuous**==:<br />$\int p(x) dx = 1$<br />$p(x) = \int p(x,y) dy$<br />$p(x)= \int p(x|y)p(y)dy$<br /> |
| **Probability distribution**<br />- It is possible to define a discrete probability distribution as a column vector<br />$p(X=x) = \begin{bmatrix} p(X=x_1)\\ \vdots \\ p(X=x_n )\end{bmatrix}$<br /><br />- The conditional probability can then be a matrix:<br />$p(x|y) = \begin{bmatrix} p(X=x_1|y_1) &\dots& p(x_1|y_m)\\ \vdots & \ddots & \vdots \\ p(X=x_n|y_1) & \dots& p(x_n| y_m) \end{bmatrix}$<br /> | **Discrete Random Variable**<br />- And the Law of Total Probabilities becomes<br />$p(x) = \underset{y}{\sum} p(x|y) p(y) = p(x|y) \cdot p(y)$<br /><br />- Note, each column of $p(x|y)$ must sum to 1<br />$\underset{x}{\sum}p(x|y)=\underset{x}{\sum}\frac{p(x,y)}{p{y}} = \frac{\underset{x}{\sum}p(y,x)}{p(y)}=\frac{p(y)}{p(y)} = 1$<br />- => Relation of joint and conditional probabilities => Total Probability = 1 |
| <br />**==Bayes Theorem==**<br />- From definition of conditional probability<br />$p(x|y) = \frac{p(x,y)}{p(y)}, \quad p(y|x) = \frac{p(x,y)}{p(x)}$<br />$p(x|y)p(y) = p(x,y) = p(y|x)p(x)$<br /><br />- Bayes Theorem Defines how to update one's beliefs about **X** ,<br />given a known (new) value of **y**<br />==$p(x|y) = \frac{p(y|x)p(x)}{p(y)}=\frac{\text{likelihood}\cdot\text{prior}}{\text{evidence}}$==<br /><br />- If **Y** is a measurement and **X** is the current vehicle state,<br />--- Bayes Theorem can be used to update the state estimate given a new measurement<br />-- **Prior**: probabilities that the vehicle is in any of the possible states<br />-- **Likelihood**: probability of getting the measurement that occurred given every possible state is the true state<br />-- **Evidence**: probability of getting the specific measurement recorded<br /> | **Gaussian Distribution** - $p(x) \sim\mathcal{N}(\mu, \sigma^2)$<br />$p(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}$<br /><br />**Multivariable** - $p(x) \sim \mathcal{N}(\mu, \Sigma)$<br /> <br />$p(x) = \frac{1}{\det{(2\pi\Sigma)}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$<br /><br /><br />**Linear Combination:**<br />$x \sim \mathcal{N}(\mu, \Sigma), \quad y = Ax + B$<br />$y \sim \mathcal{N}(A\mu+B, A\Sigma A^T)$ |









