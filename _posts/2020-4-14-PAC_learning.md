---
title: "Probably Approximately Correct "
date: 2020-04-14
tags: [machine learning, data science]
header:
  #image: "/images/NeuralNetwork.jpg"
excerpt: "Machine Learning"
mathjax: true
---



# PAC theorem

PAC stands for "probably approximately correct". As the name suggests, it relates to the probability, or confidence, of a quantity to be approximately correct. This is in essence what machine learning tries to do.  

Suppose we have a classification problem with $$N$$ classes, labelled by an integer $$\{0,1,\ldots N-1\}$$, and we are given a dataset which contains $$m$$ data-points. The data-points are characterized by $$Q$$ features, usually represented as a vector $$(f_1,f_2,\ldots,f_Q)$$, and we want to find a map $$\mathcal{G}$$ between these features and the corresponding class:

$$\mathcal{G}: (f_1,f_2,\ldots,f_Q)\rightarrow \{0,1,\ldots, N-1\}$$

This map, however, does not always exist. In this case we can only determine the class up to a certain confidence level. An example, is in image recognition where one attempts to classify pictures. Imagine that we want to determine whether the picture in hands corresponds to a dog or not. As humans, typically we find easy to identify a dog in a picture. But what if the picture was taken with a weird angle or the animal in it is actually a wolf that looks like a dog. The truth is that one cannot define exactly the class of a dog solely from the information that is stored in a picture. If we had access to other features like animal hair, body temperature, height to mass ratio, and so on, we would have been much more confident about the classification. Ultimately, having genetic information would allow us to unequivocally identify the subject (would it?).

Lets assume for the moment that such a map exists. A simple example is as follows. We want to classify points on a 2D plane as red or blue. The exact map is characterized by a circumference of radius $$R$$ cocentric with the origin of the plane, which colours points that are inside as red and outside as blue. A point $$\mathbb{x}=(x_1,x_2)$$ is sampled from an unknown distribution $$D(x)$$ and our dataset is composed of $$m$$ data-points.
