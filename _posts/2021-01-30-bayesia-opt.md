---
layout: distill
title: "Bayesian Optimization"
date: 2021-01-30
description:
tags: paramsearch bayes probml
categories: probabilistic-ml
thumbnail: assets/img/bayesian_opt/open_discussion.png
giscus_comments: true
related_posts: true
featured: true  

authors:
  - name: Quan Tran
    affiliations:
      name: RnD Department, Kyanon Digital

bibliography: bayesian-optimization.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Opening Discussion
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Levels of Optimization Problem Solving
  - name: Bayesian Optimization
  - name: Implementation

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

# Opening Discussion
<!-- ![open_discussion](/assets/img/bayesian_opt/open_discussion.png) -->
{% include figure.html path="/assets/img/bayesian_opt/open_discussion.png" class="img-fluid rounded z-depth-1" %}

# Levels of Optimization Problem Solving

In general, we can divided the way to solve optmization problem into $$3$$ levels:

* **Level 0**: Non-derivative, including some familiar approaches such as: Grid Search, Random Search, Bayesian Optimization, etc.
  - Pros: Fast computation, do not require the function to be smooth, i.e. no differentiable required, the function need continous
  - Cons: It is heuristic, not a theory-based method


* **Level 1**: First-derivative. $$∇f(x) = f(x_i)_i$$. The most popular one is Gradient Descent (Adam, Nesterov, etc.)
  - Pros: Efficient to converge in local minimum/maximum
  - Cons: Computational since it requires backpropagation

* **Level 2**: Second-derivatie. $$J(f) = f(x_i, x_j)_{ij}$$ (i.e. Newton)
  - Pros: Very effective, faster convergence than level-1 method
  - Cons: Significantly compuational


Bayesian Optimization is the one in level 0

# Bayesian Optimization

## When to use Bayesian Optimization
Supose that we are optimzing a function $$f(x, \Theta)$$
$$
\max_{\Theta}f(x,\Theta)
$$

We can consider Bayesian Optimzation if $f$ satisfy following conditions:
* $f$ has to be continuous
* $f$ is expensive to evaluate
* $f$ is a blackbox. In other words, we don't know characteristic of $f$ (convex, non-convex, derivative, etc.)

## Fundementals of Bayesian Optmization
Instead of directly optimizing $f$, we approximate it by other easier evaluation function $$P$$, and define a strategy $u$ to optimize $$P$$

* $$P$$ - **Surrogate model**: 
  - the alternative of $$f$$, easier to optmize
  - Gaussian Process, etc.
* $$u$$ - **Acquisition function**: 
  - strategy to optmize surrogate model
  - Expected Improvement (EI), Upper Condidence Bound (UCB), etc.

In this article, we focus on the most popular surrogate model - the Gaussian Process

## Gaussian Process
As mentioned, $$f$$ is black box and expensive to evaluate. Hence, the surrogate model should be an alternative that is easier to evaluate. A good intuition is to approximate $$f$$ as a distribution (normally Gaussian Distribution).  
In other words, we consider $$f$$ as a Gaussian Distribution with mean $$\mu(x)$$ and standard deviation $$\sigma^2(x)$$

$$
f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))
$$

Hence, the Gaussian Process $$P$$ has the Probability Density Function (PDF) as

$$
P(f(x)=y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp{\frac{|y - \mu(x)|^2}{2\sigma^2}}
$$

The figure below demonstrates it as a distribution approximation
<!-- ![process_step1](/img/bayesian_opt/process_step1.png) -->

Therefore, Gaussian Process makes use of Bayes Theory. Since we literally has no information about how $$f$$ looks like (*the solid red line*), we just has some initial observations (*blue circle*). **The surrogate model gives us a probabilistic estimation of $f$ as a distribution (*the area limited by dash red line*)**

From this, we need to select the next point (i.e. $$x_+$$) to sample from $$f$$. The strategy to select the next point from the observation of surrogate model is taken by **acquisition function**.

After the next point is sampled, i.e. $$\{x_+, y=f(x_+)\}$$. We continue to fit it as prior knowledge to surrogate model and continue to select next point, i.e. $$x_{++}$$. Gradually, we will have a more knowledge on how $$f$$ looks like as the belief keeps increasing.

## Acquisition function

The strategy that proposes the next sampling point is decided by acquisition function $$u$$.
Particularly, we optimize $$u$$ over surrogate model $$P$$. Note that as mentioned, those functions are easier to optimize comparing to $$f$$  
There are lots of **acquisition function**, the most popular ones are **Expected Improvement** $$EI$$, and **Upper Confidence Bound** $$UCB$$. In this article, we define $$EI$$ as the acquisition function.

Expected Improvement function is defined as
$$
EI(x) = \mathbb{E} \max (x - f(x^*))
$$

The intuition is that the chosen next sampling point should have maximal expected values comparing the best of those from observations. The best point from existing observations is denoted as
$$
x^* = \arg\max_{x_i \in D} f(x_i)
$$

Since we are optimizing acquisition function over a Gaussian Process, the more particular formula can be clarified as
$$
EI(x) = 
\begin{cases}
\displaystyle (\mu(x) - f(x^*) - \xi)\varphi(Z) + \sigma(x)\phi(Z) & \text{if} & \sigma(x) > 0 \\
\displaystyle 0 & \text{if} & \sigma(x) = 0
\end{cases}
$$

where

$$
Z = 
\begin{cases}
\displaystyle \frac{\mu(x) - f(x^*) - \xi}{\sigma(x)} & \text{if} & \sigma(x) > 0 \\
\displaystyle 0 & \text{if} & \sigma(x) = 0
\end{cases}
$$

The adjusted parameter $$\xi$$ is use to balance between *exploitation* and *exploration* of two summation terms in the above equation, respectively.

High $$\xi$$ means we lower the probability of first term (the one calculating with mean $$\mu$$, i.e. the certainty), hence, increase the probability of second term (the one calculating with variance $$\sigma$$, i.e. the uncertainty) --> we want to explore on the area of uncertainty more, and vice versa.

## The complete process

In summary, the overall process of Bayesian Optimization can be clarified as follows:
    We want to find the optimal values of objective function $$f$$ with **surrogate model** $$P$$ and **acquisition function** $$u$$. Initially, we have limited observations $$D={x_N, y_N}$$. The iteration below is how Bayesian Optimization works:  
    ---   
    1. Fit $${x_N, y_N}$$ to approximate surrogate model $$P$$  
    2. Optimize acquisition function $$u$$ over $$P$$ to sample the next point $$x_{N+} = u(x)$$  
    3. Evaluate $$\{x_{N+}\}$$ on $$f$$ to get $$y_{N+}$$  
    4. Add $$\{x_{N+}, y_{N+}\}$$ to D and repeat the process  

The detail about implementation from scratch can be found at [[4]](http://krasserm.github.io/2018/03/19/gaussian-processes/)


Below is the demonstration step-by-step:

Suppose we want to find the values that maximize the objective function below (which is unknown) with some initial observations. 
<!-- ![process_step1](/assets/img/bayesian_opt/process_step1.png) -->
{% include figure.html path="/assets/img/bayesian_opt/process_step1.png" class="img-fluid rounded z-depth-1" %}


1. Approximate a surrogate model $$P$$ (i.e. Gaussian Process). The green area is CI - Confidence Interval drawn from $$\sigma^2$$ of surrogate model, depicts the uncertainty over objective function
<!-- ![process_step2](/assets/img/bayesian_opt/process_step2.png) -->
{% include figure.html path="/assets/img/bayesian_opt/process_step2.png" class="img-fluid rounded z-depth-1" %}

2. Apply acquisition function $$u$$ over $$P$$. As a result, it proposed the next sampling point as blue circle
<!-- ![process_step3](/assets/img/bayesian_opt/process_step3.png) -->
{% include figure.html path="/assets/img/bayesian_opt/process_step3.png" class="img-fluid rounded z-depth-1" %}

3. Evaluate the proposed sampling point using objective function. We can observe that the uncertainty significantly narrow down. We repeat this process through a number of iterations
<!-- ![process_step4](/assets/img/bayesian_opt/process_step4.png) -->
{% include figure.html path="/assets/img/bayesian_opt/process_step4.png" class="img-fluid rounded z-depth-1" %}


## Discussion:

We may have a general observation about Bayesian Optimization. However, there are still some that we have resolve yet:
* Alternatively, we optimize acquisition function over surrogate model instead of objective function since it cheaper, yet it is not clear the approach to optimize acquisition function
* Beside Expected Improvement, other popular acquisition functions are UCB, Probability of Improvement

# Implementation

To simplify the implementation, we suppose that: 
* We have already known the objective function for the verification.  
* The surrogate model makes use of Gaussian Process Regression provided by `scikit-learn`  
* The process of optimizing the acquisition function will be handled by `scipy` employing *Broyden–Fletcher–Goldfarb–Shanno* algorithm[6]

We will implement
* The Bayesian Optimization process
* A method maximizing acquisition function
* An instance of acquisition function, i.e. Expected Improvement

## Set up


```python
# TO-DO: problem with scipy > 1.4, find out solution

!pip unistall scipy --y
!pip install scipy==1.4.1
```

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# assume the objective function f
def f(X, noise=0.):
    return -np.sin(6*X) - X**2 + 0.05*X + noise * np.random.randn(*X.shape)

# Initial samples
noise=0.2
X_inits = np.array([[-0.9], [1.1]])
y_inits = f(X_inits, noise)

print(X_inits.shape, y_inits.shape)

# Bounds of x
bounds = np.array([[-1.0, 2.0]])
```

    (2, 1) (2, 1)



```python
# enrich samples to draw objective function
X = np.arange(bounds[:,0], bounds[:,1], 0.01).reshape(-1,1)
y = f(X,noise=0)

print('The objective function and initial observations: ')
plt.figure(figsize=(12,8))
plt.plot(X, y, 'y--', lw=2, label='Objective function')
plt.plot(X, f(X, noise), 'gx', lw=1, alpha=0.2, label='Noisy samples')
plt.plot(X_inits, y_inits, 'kx', mew=3, label='Initial samples')
plt.legend()
plt.show()
```

    The objective function and initial observations: 



    

    


## Bayesian Optimization


```python
# !wget !wget https://raw.githubusercontent.com/krasserm/bayesian-machine-learning/dev/bayesian-optimization/bayesian_optimization_util.py
```


```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
```


```python
def process_bayesian_opt(X_samples, y_samples, gpr):
    """
    Single iteration of a Bayeisan Optimization process
    Args:
        X_samples: 
        y_samples: 
        gpr: 
        
    Returns:
        Next sampling pair {X, y}
    """
    
    # approximate GPR
    gpr.fit(X_samples, y_samples)

    # optimize acquisition over GPR to chose next sampling point X_next
    X_next = optimize_acquisition(EI, X_samples, y_samples, gpr, bounds, n_restarts=25)

    # evaluate to get y_next
    y_next = f(X_next, noise)
    
    return X_next, y_next
```

## Optimize acquisition

[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)


```python
from scipy.optimize import minimize

def optimize_acquisition(acquisition_func, X_sample, y_sample, gpr, bounds, n_restarts):
    """
    Optimize acquisition function
    Args:
        acquisition_func: 
        X_sample: 
        y_sample: 
        gpr: 
        bounds:
        n_restarts:
        
    Returns:
        proposed location
    """
    
    dim = X_sample.shape[1]
    min_value = 1
    min_x = None
    
    def minimize_objective(X):
        return -acquisition_func(X.reshape(-1,dim), X_sample, y_sample, gpr)
    
    # minimize by a popular algorithm
    for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        min_ei = minimize(minimize_objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        
        if min_ei.fun < min_value:
            min_value = min_ei.fun
            min_x = min_ei.x
    
    return min_x.reshape(-1,1)
```

## `acquisition_func`: Expected Improvement


```python
from scipy.stats import norm

def EI(X, X_sample, y_sample, gpr, xi=0.01):
    """
    Expected Improvement acquisition function
    Args:
        X
        X_sample: 
        y_sample: 
        gpr:
        xi: 
    
    Returns
        EI at X
    """
    
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    
    sigma = sigma.reshape(-1, 1)
    
    # find f(x^*)
    mu_max = np.max(mu_sample)
    
    # EI formula
    with np.errstate(divide='warn'):
        mu_dst = mu - mu_max - xi
        Z = mu_dst / sigma
        ei = mu_dst * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.] = 0.
    
    return ei
```


```python
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

n_iters = 12

X_samples = X_inits
y_samples = y_inits

plt.figure(figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iters):    
    X_next, y_next = process_bayesian_opt(X_samples, y_samples, gpr)
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    plt.subplot(n_iters, 2, 2 * i + 1)
    plot_approximation(gpr, X, y, X_samples, y_samples, X_next, show_legend=i==0)
    plt.title(f'Iteration {i+1}')

    plt.subplot(n_iters, 2, 2 * i + 2)
    plot_acquisition(X, EI(X, X_samples, y_samples, gpr).reshape(-1,1), X_next, show_legend=i==0)
    
    # Add sample to previous samples
    X_samples = np.vstack((X_samples, X_next))
    y_samples = np.vstack((y_samples, y_next))

```


    

    



```python
from bayesian_optimization_util import plot_convergence

plot_convergence(X_sample, Y_sample)
```

# References

[1] A Tutorial on Bayesian Optimization: https://arxiv.org/pdf/1807.02811.pdf  
[2] [The intuitions behind Bayesian Optimization with Gaussian Processes](https://towardsdatascience.com/the-intuitions-behind-bayesian-optimization-with-gaussian-processes-7e00fcc898a0)  
[3] [Bayesian optimization with Scikit-Optimize](https://orbi.uliege.be/bitstream/2268/226433/1/PyData%202017_%20Bayesian%20optimization%20with%20Scikit-Optimize.pdf)  
[4] [Bayesian Optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/)  
[5] [Gaussian processes](http://krasserm.github.io/2018/03/19/gaussian-processes/)  
[6] [Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)