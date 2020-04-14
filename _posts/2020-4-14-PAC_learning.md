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

Suppose we have a classification problem with $$N$$ classes, labeled by an integer $$\{0,1,\ldots N-1\}$$, and we are given a dataset which contains $$m$$ data-points. The data-points are characterized by $$Q$$ features $$(f_1,f_2,\ldots,f_Q)$$ and we want to find a map between these features and the corresponding class:
$$g: (f_1,f_2,\ldots,f_Q)\rightarrow \{0,1,\ldots, N-1\}$$
