---
title: "Probably Approximately Correct (PAC) "
date: 2020-04-14
tags: [machine learning, data science]
header:
  #teaser: /images/circle_learning_epsilon.png
  #image: "/images/NeuralNetwork.jpg"
excerpt: "One of the most important fundamental concepts of machine learning. We go over an example and later show a proof."
  #image: /images/circle_learning_epsilon.png
mathjax: true
#toc: true
classes: wide
---

## Table of Contents
1. [PAC learning](#pac)
2. [Proof](#proof)

<a name="pac"></a>
### 1. The learning problem

PAC stands for "probably approximately correct". In machine learning we want to find a hypothesis that is as close as possible to the ground truth. Since we only have access to a sample of the real distribution, the hypothesis that one builds is itself a function of the sample data, and therefore it is a random variable.  The problem that we want to solve is whether the sample error incurred in choosing a particular hypothesis  is approximately the same as the exact distribution error, within a certain confidence interval.

Suppose we have a binary classification problem (the same applies for multi-class) with classes $$y_i\in \{y_0,y_1\}$$, and we are given a training dataset $$S$$ with $$m$$ data-points. Each data-point is characterised by $$Q$$ features, and represented as a vector $$q=(q_1,q_2,\ldots,q_Q)$$. We want to find a map $$\mathcal{f}$$ between these features and the corresponding class $$y$$:

$$\mathcal{f}: (q_1,q_2,\ldots,q_Q)\rightarrow \{y_0,y_1\}$$

This map, however, does not always exist. There are problems for which we can only determine the class up to a certain confidence level. In this case we say that the learning problem is *agnostic*, while when the map exists we say that the problem is *realisable*. For example, image recognition is agnostic.

Let us assume for the moment that such a map exists. The learner chooses a set of hypothesis $$\mathcal{H}=\{h_1,\ldots,h_n\}$$ and thus introduces *bias* in the problem- a different learner may chose a different set of hypothesis. Then, in order to find the hypothesis that most accurately represents the data, the learner chooses one that has the smallest empirical risk, which is the error on the training set. That is, one tries to find the minimum of the sample loss function

$$L_S(h)=\frac{1}{m}\sum_{i=1:m}\mathbb{I}\left[h(x_i)\neq y(x_i)\right],\;h\in \mathcal{H}$$

with $$\mathbb{I}(.)$$ the Kronecker delta function. We denote the solution as $$h_S$$. The true or *generalization error* is defined instead as the unbiased average

$$\mathcal{L}(D,h)=\sum_x\mathbb{I}\left[h(x)\neq y(x)\right]D(x)$$

where $$D(x)$$ is a distribution, that the learner may or may not know. In the case of classification, the generalisation error is also the probability of misclassifying a point $$\mathcal{L}(D,h)=\mathbb{P}_{x\sim D(x)}(h(x)\neq y(x))$$.

If we choose appropriately $$\mathcal{H}$$ we may find $$\text{min}\;L_S(h_S)=0$$, for example, by memorising the data. In this case, we say that the hypothesis is *overfitting* the data. Although memorising results in zero empirical error, the solution is not very instructive because it does not give information of how well it will perform on unseen data. The solution performs very well on the data because the learner used prior knowledge to choose an hypothesis set with sufficient capacity to accommodate the entire dataset. In the above minimisation problem, one should find a solution that does well (small error) on a large number of samples rather then having a very small error in a particular sample. Overfitting solutions should be avoided as they can lead to misleading conclusions. Instead, the learner should aim at obtaining a training error that is comparable to the error obtained with different samples.

To make things practical, consider the problem of classifying points on a 2D plane as red or blue. The decision boundary is a circumference of radius $$R$$ concentric with the origin of the plane, which colours points that are inside as red and outside as blue. See figure below. The training dataset consists of $$m$$ data-points $$\mathbb{x}=(x_1,x_2)$$ sampled independently and identically distributed (i.i.d) from a distribution $$D(x)$$.

![](/images/PAC learning_1.png){: .align-center}
*Here the circumference $$R$$ denotes the ground truth which classifies points as red or blue, depending on whether they are inside or outside of the circle, respectively.*

The learning problem is to find a hypothesis $$h(x): x\rightarrow y=\{\text{blue},\text{red}\}$$ that has small error on unseen data.   

Assuming that the learner has prior knowledge of the ground truth (realisability assumption), one of the simplest algorithms is to consider the set of concentric circumferences and minimise the empirical risk. One can achieve this by drawing a decision boundary that is as close as possible to the most outward red (or inward blue data-points). This guarantees that when $$m\rightarrow \infty$$ we recover the exact decision boundary: the circumference $$R$$.  The empirical risk minimisation problem gives the solution represented in the figure below by the circumference $$R'$$. However, newly generated data-points may lie in between $$R'$$ and $$R$$, and therefore would be misclassified.

![](/images/circle_learning_epsilon.png){: .align-center}
*a) The hypothesis $$h$$ is a circumference of radius $$R'$$ concentric with the origin and it is determined by the most outward red data-point. This ensures that all training set $$S$$ is correctly classified. b) The circumference of radius $$R_{\epsilon}$$ corresponds to a hypothesis $$h_{\epsilon}$$ that has generalization error $$\mathcal{L}(D,h_{\epsilon})=\epsilon$$.*

Given that this is an overfitting solution, one has to be careful of how well it generalises. It is possible that the generalisation error is indeed small for such a solution, but one has to estimate how probable is to generate such a solution using the empirical risk.  important to estimate the probability of misclassifying data-points. To be measure how accurate the prediction is, one is interested in bounding the probability of making a bad prediction, that is,

$$\mathbb{P}_{S \sim D^m(x)}(\mathcal{L}(D,h_S)>\epsilon)<\delta \tag{1}\label{eq1}$$

Conversely, this tells us with confidence of at least $$1-\delta$$ that

$$\mathcal{L}(D,h)\leq\epsilon \tag{2}\label{eq2}$$

A *PAC learnable hypothesis* is a hypothesis for which one can put a bound on the probability of the form \eqref{eq1}.

In  the case of the circumference example, we know that $$\mathcal{L}(D,h)=\epsilon$$ happens for a radius $$R_{\epsilon}$$. Therefore any hypothesis corresponding to a radius less than $$R_{\epsilon}$$ leads to a generalization error larger than $$\epsilon$$. The probability of drawing a point and falling in the region between $$R'$$ and $$R$$ is precisely $$\epsilon$$. Therefore the probability of falling outside that region is $$1-\epsilon$$. It is then easy to see that the probability that we need equals

$$\mathbb{P}_{S \sim D^m(x)}(\mathcal{L}(D,h)>\epsilon)=(1-\epsilon)^m $$

Using the bound $$1-\epsilon<e^{-\epsilon}$$ we can choose $$\delta=e^{-\epsilon m}$$, and thus equivalently $$\epsilon=\frac{1}{m}\ln\left(\frac{1}{\delta}\right)$$. Hence using equation \eqref{eq2}, we have

$$\mathcal{L}(D,h)\leq\frac{1}{m}\ln\left(\frac{1}{\delta}\right)$$

with probability $$1-\delta$$.

<a name="proof"></a>
### 2. Finite hypothesis classes are PAC learnable

Let us assume that we have a finite hypothesis class with $$M$$ hypothesis, that is, $$\mathcal{H}_N=\{h_1,\ldots,h_N\}$$, and that this class is realisable, meaning that it contains a $$h^\star$$ for which $L_S(h^\star)=0$ for all training sets. We want to upper bound the generalisation error of a hypothesis $$h_S$$ obtained using empirical risk minimisation, that is,

$$\mathbb{P}_{x\sim D(x)}(S: L(D,h_S)>\epsilon)<\delta$$

Define a bad hypothesis as a hypothesis that has generalization error bigger than $$\epsilon$$ (does not necessarily minimize the emprirical risk). Then the set of bad hypothesis is

$$\mathcal{H}_B=\{h\in \mathcal{H}_N: L(D,h)>\epsilon\}$$

Similarly one can define the set of misleading training sets, as those that lead to a hypothesis $h_S$ with $L_S(h_S)=0$ but a generalisation error larger than $$\epsilon$$. In other words, the the datasets that can be overfitted,

$$M=\{S: h\exists \mathcal{H}_B, L_S(h)=0\}$$
