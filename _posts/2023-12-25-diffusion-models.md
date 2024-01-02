---
layout: post
title: "Diffusion Models: Fundalmentals - Part 1"
date: 2023-12-25
description:
tags: paramsearch bayes probml
categories: generative-ai
thumbnail: assets/img/
giscus_comments: true
related_posts: true
toc:
  sidebar: right
---

# What is Diffusion Models
* Motivation
    * From VAE: the same concept but gradually
    * From GAN: the same concept that generate from noise

* General concept
    * Markov Chain with 2 stages: forward and reverse
    * Forward process
    * Reverse process


* Objective function
    * Learn to generate the image from noise

* Compare 
    * to GAN: more stable in training
    * to VAE: 

## General concept

The general objective is to gradually adding noise to the original image unitl it completely becames noise, and then try to generate back to the original image from the existing noise. By learning this process, a diffusion model is expected to learn how to generate an image from a noise distribution by learning the denoising process.

Diffusion model is a Markov Chain process with two stages: forward and reverse. 

* Forward process: 
    * In the forward process, the objective is to adding noise to the image within finite steps until it completely becomes noise

* Reverse process: 
    * In the reverse process, the model learns to denoise from a noise distribution and generate back the original image within finite steps

* Objective function:
    * Similar to VAE, diffusion model optimizes the evidence lower bound (ELBO), which means we want to maximize the log-likelihood between the original image and the generated one.

# Math formulation

* We denote the real data distribution as $q(x)$
* A data point (real image) is sampled as $\mathbf{x}_0 \sim q(\mathbf{x})$
* The Markov chains is defined as $T$ finite steps

## Forward process

{% include figure.html path="/assets/img/diffusion_models/forward_process.png" class="img-fluid rounded z-depth-1" %}

* In the forward process, at timestep t, we have the noised image $x_t$ by adding noise from a distribution (normally Gaussian) to the previous image $\mathbf{x}_{t-1}$, this process is denoted as $q(\mathbf{x}_t | \mathbf{x}_{t-1})$. Formally,

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}), \beta_t \mathbf{I})
$$

where $\beta_{t}$ is a paramter to control the level of noise at timestep $t$, $\{\beta_t \in (0,1)\}_{t=1}^{T}$.

Thanks to the property of the Markov chain, we can create a tractable closed. Given $\alpha_t = 1 - \beta_t$, $\bar{\alpha_t} = \prod_{i=1}^{t} \alpha_i$, then

$$
    \mathbf{x}_t 
    = \sqrt{1 - \alpha_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon_{t-1} \\
    = \sqrt{1 - \alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar\epsilon_{t-2} \\
    = \dots \\
    = \sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t} \epsilon
$$

On the above equation, we denote $\epsilon_t \in \mathcal{N}(0, \mathbf{I})$. \bar\epsilon_{t} is a merged Gaussian distribution.

From that, the closed form of forward process is defined as 

$$
q(\mathbf{x}_t | \mathbf{x}_{0}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_{0}), (1 - \bar{\alpha}_t) \mathbf{I}
$$



## Reverse process
* Theoretically, the reverse process would be $q(\mathbf{x}_{t-1} | \mathbf{x}_{t})$.
* However, to find out the above distribution, we need to figure out the wholte data distribution, which is impractical
* The alternative is the reparameterizatin trick, sample from mean and distribution of the previous distribution. The approximation is $p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)$.
* Because $q(\mathbf{x}_{t-1} | \mathbf{x}_{t})$ is a Gaussian distribution, with a small $\beta_t$, $p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)$. is also a Gaussian distribution, so we can sample using the above clarification.

Formally,
$$
p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t}; \mathbf{\mu}_{\theta}(\mathbf{x}_t, t), \mathbf{\sum}_{\theta}(\mathbf{x}_t, t))
$$


{% include figure.html path="/assets/img/diffusion_models/reverse_process.png" class="img-fluid rounded z-depth-1" %}

## Objective function
We optimize the ELBO function on the negative log-likelihod function.
$$
\mathbb{E}[-\log p_{\theta}(\mathbf{x_0})] 

\leq 

\mathbb{E_q}[-\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}]

\\

= \mathbb{E_q}[-\log p(\mathbf{x}_T) 

- \sum_{t \geq 1} 

\frac{ p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)} 

{q(\mathbf{x}_{t} | \mathbf{x}_{t-1})} ]

\\

\eqqcolon L

$$

Derive the above function, we get

$$

L \coloneqq 

\mathbb{E_q} 

\left[

   \underbrace{D_{KL} \left( q(\mathbf{x}_T | \mathbf{x}_0) \enspace || \enspace p_{\theta}(\mathbf{x}_T) \right) _ {L_T}}
   _{L_T}

+   \underbrace{\sum\limits_{t \geq 1} D_{KL} \left( q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_{0}) \enspace || \enspace p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t}) \right) }
    _{L_{t-1}}

        - \underbrace{\log p_{\theta} (\mathbf{x}_0 | \mathbf{x}_1)}
        _{L_0}

\right]

$$

In the above training loss, we observer there are three parts:
* $L_T$ is a constant, and can be ignored during the training since $q$ has no learnable parameters and $\mathbf{x}_t$ is a Gaussian noise
* $L_0$ is a reconstruction term and is learned using a seperate decoder following $\mathcal{N} \sim (\mathbf{x}_0; \mu_\theta(\mathbf{x}_1, 1), \Sigma_\theta (\mathbf{x}_1, 1) )$
* $L_{t-1}$ is a learnable parameters and we focus on how to learn this subloss function.

In $L_{t-1}$, the term $q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_{0})$ has not been defined. In words, it means we wish to denoise the image from previous noisy one $\mathbf{x}_{t}$ and it is also conditioned on the original image $\mathbf{x}_0$. As a result, 

$$
    q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_{0}) \sim 
            \mathcal{N}(\mathbf{x}_{t-1}, 
            \tilde\mu_t(\mathbf{x}_{t}, \mathbf{x}_0),
            \tilde\beta_{t} \mathbf{I}) 
$$

with

$$
    \tilde\beta_t = \frac{1- \bar\alpha_{t-1}}{1 - \bar\alpha_t} \cdot \beta_t
$$

Using the Bayes rules, $\tilde\mu_t(\mathbf{x}_t, \mathbf{x}_0)$ can be derived into

$$
    \tilde\mu_t(\mathbf{x}_t, \mathbf{x}_0) = 
        \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} \mathbf{x}_0
        + 
        \frac{\sqrt\alpha_t (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} \mathbf{x}_t
$$

However, it still depends on two variables, $\mathbf{x}_0$ and $\mathbf{x}_t$, we want to transform it to only depends on one variable. Because we have a tractable closed-form of $\mathbf{x}_0$ and $\mathbf{x}_t$ and $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$, the above equation would become

$$
   
     \tilde\mu_t(\mathbf{x}_t) =
        \frac{1}{\sqrt{\alpha_t}} (
            \mathbf{x}_{t} - 
            \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_t
        )

$$



Recall the $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ derive formula, we can have the same transformation with a learnable $\epsilon_\theta(\mathbf{x}_t, t)$ for $\mu_\theta(\mathbf{x}_t, t)$

$$
    \mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} (
            \mathbf{x}_{t} - 
            \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(\mathbf{x}_t, t)
        )
$$

From the above equation, we observe that the generated $\mathbf{x}_t$ depends only on a trainable variable, which is $\epsilon_\theta(\mathbf{x}_t, t)$, at timestep $t$. The problem turns out to predict the noise of the generated image for every step $t$ in the denoising process. As a result, we define a neural network to predict $\epsilon_\theta(\mathbf{x}_t, t)$

Applying the above derive into the $L_{t-1}$, now the objective is to minimize the difference between the current noise and the predicted noise

$$
    L_{t} = \mathbb{E}_{\mathbf{x}_0, \epsilon}
        \left[
            \frac{1}{2 || \Sigma_\theta (\mathbf{x}_t, t) ||_2^2}
            ||  \tilde\mu_t(\mathbf{x}_t, \mathbf{x}_0) -  \mu_\theta(\mathbf{x}_t, t) ||^2
        \right]

        \\

        = \mathbb{E}_{\mathbf{x}_0, \epsilon}
        \left[
            \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar\alpha_t) || \Sigma_\theta ||_2^2}
            || \epsilon_t - \epsilon_\theta(\sqrt{\alpha_t} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t} \epsilon_t, t) ||^2
        \right]
$$

By discarding the regulaization term, the authors come with a simpler version

$$
    L_{t}^{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \epsilon}
        \left[
            || \epsilon_t - \epsilon_\theta(\sqrt{\alpha_t} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t} \epsilon_t, t) ||^2
        \right]
$$

Since we ignore the other parts of the overall loss function and with the simple version, the final loss function is

$$
    L_{\text{simple}} = L_{t}^{\text{simple}} + C
$$

where $C$ is a constant

### *Explanation for the impractical of $q(\mathbf{x}_{t-1} | \mathbf{x}_{t})$.

### *Explanation for the reparameterization trick

### *Bayes rule to expand the $q(\mathbf{x}_{t} | \mathbf{x}_{t-1}, \mathbf{x}_{0})$

# Overall training process
The process of developing diffusion model consitst of training and sampling.

## Training
For each training step:
* Sample an original image from the dataset: $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
* Sample range of timestep $t$ in range (1, $T$), for example, sampling with a uniform: $t \sim \text{Uniform}(\{1, \dots, T\})$
* Sample noise: $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
* Calculate the gradient folllowing the loss function:
$ \Delta_\theta || \epsilon - \epsilon_\theta(\sqrt{\alpha_t} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t} \epsilon_t, t)||^2 $

## Sampling
The sampling process is when we want to generate the image from the noise distribution.

* First, we generate the noise: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
* For each timestep from $T$ to $1$
    * Sample embedding from the noise distribution: 
    $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    * Calculate the denoising version at timestep $t-1$ using the reparameterization trick:
    $$ 
    \mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)

    \\

     \mathbf{x}_{t-1} = 
        \frac{1}{\sqrt{\alpha_t}}
            \left(
                \mathbf{x}_{t-1} - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t} \epsilon_\theta(\mathbf{x}_t, t)} + \sigma_t \mathbf{z}
            \right)
    $$

* Get $\mathbf{x}_0$

The two process are summarized as follow Algorithms

{% include figure.html path="/assets/img/diffusion_models/training_sampling_algorithms.png" class="img-fluid rounded z-depth-1" %}
# Model backbone

# Conditional generation

# Beyond conventional diffusion models
