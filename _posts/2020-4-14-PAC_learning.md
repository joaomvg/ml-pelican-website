---
title: "Probably Approximately Correct (PAC) "
date: 2020-04-14
tags: [machine learning, data science]
header:
  #image: "/images/NeuralNetwork.jpg"
excerpt: "Machine Learning"
mathjax: true
toc: true
---



### 1. The learning problem

PAC stands for "probably approximately correct". As the name suggests, it relates to the probability, or confidence, of a quantity to be approximately correct. This is in essence what machine learning tries to do.  

Suppose we have a classification problem with $$N$$ classes $$y_i\in {y_0,y_1,\ldots,y_{N-1}}$$, and we are given a training dataset $$S$$ with $$m$$ data-points. Each data-point is characterised by $$Q$$ features, usually represented as a vector $$(f_1,f_2,\ldots,f_Q)$$, and we want to find a map $$\mathcal{G}$$ between these features and the corresponding class $$y$$:

$$\mathcal{G}: (f_1,f_2,\ldots,f_Q)\rightarrow y=\{y_0,y_1,\ldots, y_{N-1}\}$$

This map, however, does not always exist. In this case we can only determine the class up to a certain confidence level. An example, is in image recognition where one attempts to classify pictures. Imagine that we want to determine whether the picture in hands corresponds to a dog or not. As humans, typically we find easy to identify a dog in a picture. But what if the picture was taken with a weird angle or the animal in it is actually a wolf that looks like a dog. The truth is that one cannot define exactly the class of a dog solely from the information that is stored in a picture. If we had access to other features like animal hair, body temperature, height to mass ratio, and so on, we would have been much more confident about the classification. Ultimately, having genetic information would allow us to unequivocally identify the subject (would it?).

Let us assume for the moment that such a map exists. Consider the problem of classifying points on a 2D plane as red or blue. The exact map is characterised by a circumference of radius $$R$$ concentric with the origin of the plane, which colours points that are inside as red and outside as blue. See figure below. The training dataset consists of $$m$$ data-points $$\mathbb{x}=(x_1,x_2)$$ sampled independently and identically distributed (i.i.d) from a distribution $$D(x)$$. In most instances, we do not know this distribution.

![](/images/PAC learning_1.png){: .align-center}
*Here the circumference $$R$$ denotes the ground truth which classifies points as red or blue, depending on whether they are inside or outside of the circle, respectively.*

The learning problem is to find a hypothesis $$h(x): x\rightarrow y$$ that has small error on unseen data.
The *empirical error* or *training error* is given by a loss function calculated on the training dataset $$S$$ and defined as follows:

$$\mathcal{L}_S(h)=\frac{1}{m}\sum_{i=1:m}\mathbb{I}\left[h(x_i)\neq y(x_i)\right]$$

which is just the ratio of the number of misclassified points over the total number of data-points. Here the function $$\mathbb{I}(.)$$ is the Kronecker delta function which gives one when the condition is satisfied and zero otherwise. The true error or *generalization error* is the unbiased estimator

$$\mathcal{L}(D,h)=\sum_x\mathbb{I}\left[h(x)\neq y(x)\right]D(x)$$

and equals the probability of misclassifying a point:

$$\mathcal{L}(D,h)=\mathbb{P}_{x\sim D(x)}(h(x)\neq y(x))$$

When a hypotheses $$h(x)$$ that has zero empirical error one says that it *overfits* the data. This can be achieved, for example, by memorising all the training data. While this works well on the training set, it may lead to very misleading predictions on unseen data. The problem, explained simply, is due essentially to the following reasons: we may be overfitting on data that is not representative and so the hypotheses will generalise poorly, and secondly memorising all the data requires a very complex function $$h(x)$$ which leads to a prediction with high variance.   

One of the simplest algorithms is to draw a decision boundary that is as close as possible to the most outward red (inward blue data-points). Note that we have chosen a set of hypothesis $$\mathcal{H}$$ that contains the ground truth: the set of concentric circumferences. This is called the *realizability assumption*. This guarantees that when $$m\rightarrow \infty$$ we recover the exact decision boundary: the circumference $$R$$.  This choice of $$h\in \mathcal{H}$$, as represented in the figure below by the circumference $$R'$$, ensures that all the points in the sample data are correctly classified. However newly generated data samples may lie in between $$R'$$ and $$R$$, and therefore would be misclassified.

![](/images/circle_learning_epsilon.png){: .align-center}
*a) The hypothesis $$h$$ is a circumference of radius $$R'$$ concentric with the origin and it is determined by the most outward red data-point. This ensures that all training set $$S$$ is correctly classified. b) The circumference of radius $$R_{\epsilon}$$ corresponds to a hypothesis $$h_{\epsilon}$$ that has generalization error $$\mathcal{L}(D,h_{\epsilon})=\epsilon$$.*

Since overfitting can lead to very erroneous predictions, it is important to estimate the chance of that happening. Suppose we have a bound on this probability of the form

$$\mathbb{P}_{S \sim D^m(x)}(\mathcal{L}(D,h)>\epsilon)<\delta \tag{1}\label{eq1}$$

Note that the probability is calculated against drawing a sample $$S$$ with $$m$$ data-points and $$h$$ is the overfitting hypothesis that results from this sample. Conversely, we know  

*with confidence of at least $$1-\delta$$ that $$\mathcal{L}(D,h)\leq\epsilon\tag{2}\label{eq2}$$.*

A *PAC learnable hypothesis* is a hypothesis for which one can put a bound on the probability of the form \eqref{eq1}.

In  the case of the circumference example, we know that $$\mathcal{L}(D,h)=\epsilon$$ happens for a radius $$R_{\epsilon}$$. Therefore any hypothesis corresponding to a radius less than $$R_{\epsilon}$$ leads to a generalization error larger than $$\epsilon$$. The probability of drawing a point and falling in the region between $$R'$$ and $$R$$ is precisely $$\epsilon$$. Therefore the probability of falling outside that region is $$1-\epsilon$$. It is then easy to see that the probability that we need equals

$$\mathbb{P}_{S \sim D^m(x)}(\mathcal{L}(D,h)>\epsilon)=(1-\epsilon)^m $$

Using the bound $$1-\epsilon<e^{-\epsilon}$$ we can choose $$\delta=e^{-\epsilon m}$$, and thus equivalently $$\epsilon=\frac{1}{m}\ln\left(\frac{1}{\delta}\right)$$. Hence using equation \eqref{eq2}, we have

$$\mathcal{L}(D,h)\leq\frac{1}{m}\ln\left(\frac{1}{\delta}\right)$$

with probability $$1-\delta$$.

### 2. Finite hypothesis classes are PAC learnable

asdasd
