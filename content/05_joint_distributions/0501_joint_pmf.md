# Joint PMF

```{prf:definition} Joint PMF
:label: def_joint_pmf

Let $X$ and $Y$ be two discrete random variables with sample spaces $\S_X$ and $\S_Y$ respectively. 
The ***joint probability mass function (joint PMF)*** of $X$ and $Y$ is the function

$$
\pmfjointxy(x, y) = \P \lsq X = x \wedge Y = y \rsq = \P \lsq (\omega, \xi) ~ \lvert ~ X(\omega) = x \wedge Y(\xi) = y \rsq
$$

where $\P$ is the probability function defined over the probability space $\pspace$.
```

```{figure} ../assets/fig5.4.png
---
name: fig_joint_pmf
---
Joint PMF of $X$ and $Y$. Image Credit: {cite}`chan_2021`
```

```{prf:definition} Joint PDF
:label: def_joint_pdf

Let $X$ and $Y$ be two continuous random variables with sample spaces $\S_X$ and $\S_Y$ 
respectively.
The joint PDF of $X$ and $Y$ is a function $\pdfjointxy(x, y)$ that can be integrated to yield a 
probability

$$
\P \lsq \eventA \rsq = \int_{\eventA} \pdfjointxy(x, y) \, dx \, dy
$$ (eq_joint_pdf)

for any event $\eventA \subseteq \S_X \times \S_Y$.
```

```{prf:theorem} Joint PMF and Joint PDF
:label: thm_joint_pmf_pdf

Let $\S$ = $\S_X \times \S_Y$.
All joint PMFs and joint PDFs satisfy 

$$ 
\sum_{(x, y) \in \S} \pmfjointxy(x, y) = 1 \quad \text{and} \quad \int_{\S} \pdfjointxy(x, y) \, dx \, dy = 1
$$ (eq_joint_pmf_pdf)
```

```{prf:definition} Marginal PMF
:label: def_marginal_pmf

The marginal PMF is defined as 

$$
p_X(x) = \sum_{y \in \S_Y} \pmfjointxy(x, y) \quad \text{and} \quad p_Y(y) = \sum_{x \in \S_X} \pmfjointxy(x, y)
$$

and the marginal PDF is defined as 

$$
f_X(x) = \int_{\S_Y} \pdfjointxy(x, y) \, dy \quad \text{and} \quad f_Y(y) = \int_{\S_X} \pdfjointxy(x, y) \, dx
$$
```

```{prf:definition} Independent random variables
:label: def_independent

Random variables $X$ and $Y$ are ***independent*** if and only if

$$
\pmfjointxy(x, y) = p_X(x) p_Y(y) \quad \text{or} \quad \pdfjointxy(x, y) = f_X(x) f_Y(y)
$$
```

```{prf:definition} Independence for N random variables
:label: def_independent_n

A sequence of random variables $X_1$,...,$X_N$ is **independent** if and only 
if their joint PDF (or joint PMF) can be factorized. 

$$ 
f_{X_1}, \ldots, X_n(x_1, \ldots, x_n) = \prod_{n=1}^N f_{X_n}(x_n)
$$
```

```{prf:definition} Independent and Identically Distributed (i.i.d.)
:label: def_iid

A collection of random variables $X_1$, ..., $X_N$ is called *independent and 
identically distributed (i.i.d.)* if 

- All $X_1$, ..., $X_N$ are independent; and
- All $X_1$, ..., $X_N$ have the same distribution,
i.e. $f_{X_1}(x) = \cdots = f_{X_N}(x)$
```

```{prf:remark} Why is $\iid$ so important?
:label: rem_iid
- If a set of random variables are $\iid$, then the joint PDF can be written as 
a products of PDFs. 
- Integrating a joint PDF is difficult. Integrating a product of PDFs is 
much easier. 
```

```{prf:example} Gaussian $\iid$
:label: ex_gaussian_iid

Let $X_1, X_2, \ldots, X_N$ be a sequence of $\iid$ Gaussian random variables
where each $X_i$ has a PDF

$$
f_{X_i}(x) = \frac{1}{\sqrt{2 \pi}} \exp \lsq - \frac{x^2}{2} \rsq
$$

The joint PDF of $X_1, X_2, \ldots, X_N$ is

$$
\begin{aligned}
f_{X_1, \ldots, X_N}(x_1, \ldots, x_N) &= \prod_{i=1}^N \lsq \frac{1}{\sqrt{2 \pi}} \exp \lsq - \frac{x_i^2}{2} \rsq \rsq \\ 
                                       &= (\frac{1}{\sqrt{2 \pi}})^N \exp \lsq - \sum_{i=1}^N \frac{x_i^2}{2} \rsq \\
\end{aligned}
$$

which is a function depending not on the individual values of $x_1, x_2, \ldots, 
x_N$, but on the sum $\sum_{i=1}^N x_i^2$. So we have "compressed" an
N-dimensional function into a 1D function.
```

```{prf:example} Gaussian $\iid$ (cont.)
:label: ex_gaussian_iid_cont

Let $\theta$ be a deterministic number that was sent through a noisy channel. 
We model the noise as an additive $\gaussian$ random variable with mean 0 and 
variance $\sigma^2$. Supposing we have observed measurements 
$X_i$ = $\theta$ + $W_i$, for i = 1, $\ldots$ , N, where 
$W_i$ ~ $\gaussian$(0, $\sigma^2$), then the PDF of each $X_i$ is

$$
f_{X_i}(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \lsq - \frac{(x - \theta)^2}{2 \sigma^2} \rsq
$$ 

Thus the joint PDF of ($X_1, X_2, \ldots, X_N$) is

$$
\begin{aligned}
f_{X_1, \ldots, X_N}(x_1, \ldots, x_N) &= \prod_{i=1}^N \lsq \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \lsq - \frac{(x_i - \theta)^2}{2 \sigma^2} \rsq \rsq \\
                                       &= \lsq \frac{1}{\sqrt{2 \pi \sigma^2}} \rsq^N \exp \lsq - \sum_{i=1}^N \frac{(x_i - \theta)^2}{2 \sigma^2} \rsq \\
\end{aligned}
$$

Essentially, this joint PDF tells us the probability density of seeing sample 
data $x_1, \ldots, x_N$.
```
