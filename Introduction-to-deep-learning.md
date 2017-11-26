---
title: "Introduction to Deep Learning"
author: Ray Cai
date: October 7, 2017
output: pdf_document
markdown:
  image_dir: figures_generated
  absolute_image_path: false
presentation:
    width: 1280
    height: 800
    slideNumber: true
    center: false
---

<style type="text/css">
  .reveal section img {
    background-color: white !important;
    max-height: 600px !important;
  }
  .reveal h2 {
    text-align: left;
  }
  .reveal p {
    text-align: left;
  }
  .reveal ul {
    display: block;
  }
  .reveal ol {
    display: block;
  }
  p.diagram {
    color: #abb2bf;
    background-color: #31363f !important;
    border: #4b5362;
    border-radius: 3px;
  }
</style>

<!-- slide -->
## Introduction to Deep Learning

> by **Ray Cai**
> from **PMC, ISG, ICG**

<!-- slide -->
## Agenda

1. Application Areas
2. Approaches

<!-- slide -->
## Application Areas

* Speech Recognition
* Image Recognition
* Natural Language Processing
* Visual Art Processing
* Bioinformatics

<!-- slide -->
## Speech Recognition

![](figures/google-speech-recognition-error-rate.png)

* [Google Speech API](https://cloud.google.com/speech/)
<!-- slide -->
## Speech Recognition

![](figures/google-speech-api-pricing.png)

**$1.44 per Hour**

* [Google Speech API](https://cloud.google.com/speech/)

<!-- slide -->
## Image Recognition

![](figures/google-vision-api-emotions.png)

* [Detect Faces and Emotions](https://developers.google.com/vision/)
<!-- slide -->
## Image Recognition

![](figures/google-vision-api-activity.png)

* [Google Cloud Vision API enters Beta, open to all to try!](https://cloudplatform.googleblog.com/2016/02/Google-Cloud-Vision-API-enters-beta-open-to-all-to-try.html)

<!-- slide -->
## Image Recognition

![](figures/google-vision-api-logo.png)

* [Google Cloud Vision API enters Beta, open to all to try!](https://cloudplatform.googleblog.com/2016/02/Google-Cloud-Vision-API-enters-beta-open-to-all-to-try.html)

<!-- slide -->
## Natural Language Processing

![](figures/google-lanaguage-entities.png)

* [Google Natural Language API](https://cloud.google.com/natural-language/)

<!-- slide -->
## Natural Language Processing

![](figures/google-language-syntax.png)

* [Google Natural Language API](https://cloud.google.com/natural-language/)

<!-- slide -->
## Visual Art Processing

![](figures/style-transfer-1.png)

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
<!-- slide -->
## Visual Art Processing

![](figures/style-transfer-2.png)

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
<!-- slide -->
## Visual Art Processing

![](figures/style-transfer-3.png)

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

<!-- slide -->
## Approaches

<!-- slide -->
## Deep Learning

* Machine Learning
  * Support Vector Machine
  * Artificial Neural Network
    * **Deep learning**
  * Decision Tree
  * ...

<!-- slide -->
## Artificial Neural Network

<!-- slide -->
## Biological Neuron

![](figures/Structure-Of-Neurons-In-Brain.jpg)

<!-- slide -->
## Mathematical Neuron

![](figures/Mathenmatic-Neuron.png)

$$
f = \psi(\sum(X \times W + b))
$$

<!-- slide -->
## Neural Network

![](figures/neural_network.png) 

<!-- slide -->
## Handwritten Digits Classification

![](figures/6a54f12d0f63c9bc.png)

<!-- slide -->
## 1-Layer Neural Network

![](figures/d5222c6e3d15770a.png)

<!-- slide -->
## 1-Layer Neural Network - Weight

$$
W = 
\begin{bmatrix}
w_{0,0} & w_{0,1} & w_{0,2} & w_{0,3} & ... & w_{0,9} \\
w_{1,0} & w_{1,1} & w_{1,2} & w_{1,3} & ... & w_{1,9} \\
w_{2,0} & w_{2,1} & w_{2,2} & w_{2,3} & ... & w_{2,9} \\
w_{3,0} & w_{3,1} & w_{3,2} & w_{3,3} & ... & w_{3,9} \\
... \\
w_{783,0} & w_{783,1} & w_{783,2} & w_{783,3} & ... & w_{783,9} 
\end{bmatrix}
$$

<!-- slide -->
## 1-Layer Neural Network - Activation Function

$$
\psi = softmax(L_n) = \frac{e^{L_n}}{||e^L||}
$$

<!-- slide -->
## 1- Layer Neural Network - Formula

![](figures/206327168bc85294.png)

<!-- slide -->
## 1- Layer Neural Network - Loss Function

![](figures/1d8fc59e6a674f1c.png)

<!-- slide -->
## Mathematical Problem

**Known:**

$$
Y = softmax(X.W+b)
$$

$$
L = -\Sigma{Y'_i.log(Y_i)}
$$

**Adjust $W$ and $b$ minimise $L$**

<!-- slide -->
## Gradient Descent

![](figures/34e9e76c7715b719.png)

<!-- slide -->
## Accuracy

![](figures/e102f513bec53e08.png)

<!-- slide -->
## Deep Neural Network

![](figures/77bc41f211c9fb29.png)

<!-- slide -->
## Mathematical Problem

$$
T_1 = sigmoid(X.W_1+b_1) \\
T_2 = sigmoid(T_1.W_2+b_2) \\
T_3 = sigmoid(T_2.W_2+b_2) \\
T_4 = sigmoid(T_3.W_3+b_3) \\
Y = sigmoid(T_4.W_4+b_4)
$$

$$
L = -\Sigma{Y'_i.log(Y_i)}
$$

<!-- slide -->
## Accuracy
![](figures/dbbf4c8edae90438.png)

<!-- slide -->
## More Complex Neural Network

* Convolutional networks

<!-- slide -->
## Convolutional Networks

![](figures/53c160301db12a6e.png)

<!-- slide -->
## Accuracy

![](figures/convolutional-result.png)

<!-- slide -->
## Reference

1. [TensorFlow and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist)
2. [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning)

<!-- slide -->
## Q&A

<!-- slide -->
## Thank You