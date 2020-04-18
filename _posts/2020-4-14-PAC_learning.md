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

Suppose we have a classification problem with $$N$$ classes, labelled by an integer $$\{0,1,\ldots N-1\}$$, and we are given a dataset which contains $$m$$ data-points. The data-points are characterized by $$Q$$ features, usually represented as a vector $$(f_1,f_2,\ldots,f_Q)$$, and we want to find a map $$\mathcal{G}$$ between these features and the corresponding class $$y$$:

$$\mathcal{G}: (f_1,f_2,\ldots,f_Q)\rightarrow y=\{0,1,\ldots, N-1\}$$

This map, however, does not always exist. In this case we can only determine the class up to a certain confidence level. An example, is in image recognition where one attempts to classify pictures. Imagine that we want to determine whether the picture in hands corresponds to a dog or not. As humans, typically we find easy to identify a dog in a picture. But what if the picture was taken with a weird angle or the animal in it is actually a wolf that looks like a dog. The truth is that one cannot define exactly the class of a dog solely from the information that is stored in a picture. If we had access to other features like animal hair, body temperature, height to mass ratio, and so on, we would have been much more confident about the classification. Ultimately, having genetic information would allow us to unequivocally identify the subject (would it?).

Let us assume for the moment that such a map exists. Consider the problem of classifying points on a 2D plane as red or blue. The exact map is characterised by a circumference of radius $$R$$ concentric with the origin of the plane, which colours points that are inside as red and outside as blue. See figure below. The training dataset consists of $$m$$ data-points $$\mathbb{x}=(x_1,x_2)$$ sampled independently and identically distributed (i.i.d) from a distribution $$D(x)$$. In most instances, we do not know this distribution.

The learning problem is to find an hypotheses $$h(x): x\rightarrow y$$ that has small error on unseen data. The *empirical error* or *training error* is calculated by a loss function calculated on the training dataset $$S$$ and defined as follows:

$$\mathcal{L}_S(h)=\frac{1}{m}\sum_{i=1:m}\mathbb{I}\left[h(x_i)\neq y(x_i)\right]$$

which is just the ratio of the number of misclassified points over the total number of data-points. Here the function $$\mathbb{I}(.)$$ is the Kronecker delta function which gives one when the condition is satisfied and zero otherwise. The true error or *generalization error* is the unbiased estimator

$$\mathcal{L}(D,h)=\sum_x\mathbb{I}\left[h(x)\neq y(x)\right]D(x)$$

and equals the probability of a point to be misclassified:

$$\mathcal{L}(D,h)=\mathbb{P}(h(x)\neq y(x))$$

One of the simplest algorithms is to draw a decision boundary, call it $$\mathcal{C}$$, that is as close as possible to the most outward red or inward blue data-points. Needless to say, when $$m\rightarrow \infty$$ we recover the exact decision boundary: the circumference. This ensures that all the points in the sample data are correctly classified. The problem however is that if we draw more data samples we can generate points that lie in between $$\mathcal{C}$$ and the circumference of radius $$R$$, and would therefore be misclassified.  Simply memorising the data can lead to very erroneous outcomes- this is also known as overfitting.


![](/images/PAC learning.png){: .align-center}
*Learning with circles. Here the circumference $$R$$ denotes the ground truth which classifies points as red or blue. The hypotheses $$h$$ correctly classifies all the training points and we choose it to pass by the most outward red training point.*
