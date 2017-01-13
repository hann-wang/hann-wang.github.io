---
layout:     post
title:      "Literature Summary for GAN-Based Visual Recognition"
subtitle:   "R-CNN, Fast R-CNN, Faster R-CNN, GAN, CGAN, LAGAN, CatGAN, Feature Matching..."
date:       2017-01-13
author:     "Hann"
header-img: "img/post-bg-ann.jpg"
tags:
    - Deep Learning
    - Visual Recognition
---

> This post might be translated into Chinese later.
> 
> 后期本文将可能被翻译为中文。


## Introduction

Visual recognition, mainly referring to image classification and object detection, is one of the most popular research topics in computer vision field. Traditional algorithms for this task usually involve the use of SIFT and HOG, which have achieved rather minor improvement since 2010. In 2012, Krizhevsky et al. [krizhevsky2012imagenet] rediscovered CNNs and showed substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge(ILSVRC). Since then, more approaches have been proposed and proved to be successful for image classification. Unlike image classification, detection requires localizing objects within an image. R-CNN [girshick2014rich] proposed by R. Girshick transforms a detection problem into "recognition using regions" and thus removes the gap between classification and detection. The improvements of R-CNN, also known as Fast R-CNN [girshick2015fast] and Faster R-CNN [ren2015faster] now lay the foundation for most of the modern object detection schemes.

So far most of the CNN-based visual recognition networks are discriminative models, mapping a high-dimensional input to a class label. Their successes are primarily based on the backpropagation and dropout algorithms, where a particularly well-behaved gradient can be calculated according to the loss function. In 2014, I. J. Goodfellow [goodfellow2014generative] suggested Generative Adversarial Networks (GANs), involving both generative and discriminative models and pointed out a new structure for deep learning networks. Many significant works [mirza2014conditional], [denton2015deep], [radford2015unsupervised], [springenberg2015unsupervised], [salimans2016improved] relating to GANs have been done recently and arouse widespread interests for they have provided an attractive alternative to maximum likelihood techniques.

Some of the papers mentioned above are summarized in this work.

## R-CNNs

### R-CNN

CNNs were heavily used during the 1990s before the rise of support vector machines. In 2012, Krizhevsky et al. [krizhevsky2012imagenet] rekindled interests in CNNs by showing substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). The CNN classification results were extended to object detection by R. Girshick [girshick2014rich], who bridged the gap between image classification and object detection by introducing R-CNN (Regions with CNN features). At test time, R-CNN generates around 2000 category-independent region proposals for the input image, extracts a fixed-length feature vector from each proposal using a CNN, and then classifies each region with category-specific linear SVMs. The CNN model is pre-trained on ILSVRC2012 and domain-specific fine-tuned on Pascal VOC 2010 and 2012. Once features are extracted and training labels are applied, one linear SVM per class is optimized.
	
### Fast R-CNN

Comparing to image classification, object detection is faced with two primary challenges: First, numerous region proposals must be processed; Second, these proposals must be refined to achieve precise localization. Shortly after R-CNN published, R. Girshick changed it into Fast-RCNN [girshick2015fast], which first processes the who image with several convolution layers, and then use a ROI-pooling layer to extract a fix-length vector from the feature map for each ROI. Each feature vector is fed into a sequence of fully-connected layers that finally branch into two sibling output layers: One that produces softmax probability estimates over *K* object classes plus one background class, while the other performs bounding box regression.
	
### Faster R-CNN

Fast R-CNN, achieves near real-time rates using very deep networks, when ignoring the time spent on region proposals which is done outside the network. Faster R-CNN [ren2015faster] proposed by S. Ren et al. tries to perform region proposal using a deep convolution network (Region Proposal Network, RPN) so that it can be embedded into object detection nets and leads to real-time performance. RPN takes the feature map computed by convolution layers in Fast R-CNN and slides a small network over it while each sliding window is mapped into a lower-dimensional feature and then fed into the two sibling fully-connected layers in Fast R-CNN. At each sliding-window location, multiple region proposals are predicted with a maximum limit *k*. The *k* proposals are relative to *k* anchor boxes centered at the sliding window and associated with a scale and aspect ratio. 

## GANs

### GAN

All the CNN-based visual recognition networks mentioned in the former section are all discriminative models trained by back-propagation algorithm which computes gradients according to loss functions. However, I. J. Goodfellow et al. [goodfellow2014generative] proposed adversarial nets which have no loss functions but only a minimax game. A generator tries to approximate the probability distribution of training dataset and samples from the model distribution, while a discriminator learns to determine whether a sample is from the model distribution or the data distribution. Competition in this game drives both the generator and the discriminator to improve themselves. This framework can yield specific training algorithms for many kinds of model and optimization algorithm.
	
### CGAN

M. Mirza and S. Osindero [mirza2014conditional] extended the original GAN into Conditional GAN (CGAN), where both the generator and the discriminator take additional information to direct the data generation and discrimination process. Such conditioning could be based on class labels, on some part of blur data, or even on data from different modality. Two experiments are performed in their paper: One is trained on MNIST dataset conditioned on one-hot coded class labels and the other is on YFCC100M dataset to generate a distribution of tag-vectors conditioned on image features.
	
### LAPGAN

Facebook AI Lab started a project called *Eyescream* based on I. J. Goodfellow's GAN but they found it not suitable for generating images of large size, so they constructed a Laplacian pyramid [denton2015deep] for each training image *I*. During training time at each level, the generator takes a blur and downsampled image as well as a noise vector as input and generate a high-pass image while the discriminator computes the probability of it being real or generated. During testing time, the pyramid is reversed so that a small image is generated in the first layer and high-pass images are generated in the rest to be added to the upsampled image from the previous layer. Samples from this model show higher log-likelihood than from original GAN.
	
### DCGAN

A. Radford et al. [radford2015unsupervised] propose and evaluate a set of constraints on the architectural topology of Deep Convolutional GANs (GCGANs) that make them stable to train in most settings:
	
* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
* Use batch-norm in both generator and discriminator.
* Remove fully-connected hidden layers for deeper architectures.
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
* Use LeakyReLU activation in discriminator for all layers.  

The trained discriminators shows competitive performance for image classification tasks with other unsupervised algorithms.
	
### CatGAN

J. T. Springenberg [springenberg2015unsupervised] argues that the discriminator of DCGAN is learning the features which are not yet correctly modeled by the generator. In turn, these features need not necessarily align with our classification purpose. Under the worst circumstances the discriminator of DCGAN could be detecting noise in the generated data. Instead asking the discriminator to distinguish the generated data from the natural one, Springenberg requires it to assign all examples to one of *K* categories, while staying uncertain for samples from the generator. Experiments on MNIST and CIFAR-10 show that CatGAN is comparable to the state-of-the-art semi-supervised learning methods.
	
### Feature Matching

I. J. Goodfellow's original paper [goodfellow2014generative] suggests that the GANs could be trained using back-propagation algorithm. However, T. Salimans [salimans2016improved] argue that it is rather hard to find Nash equilibrium in a non-convex game with continuous high-dimensional parameters using gradient descent techniques. In order to encourage the convergence of GANs, they provide five means: 
	
* **Feature Matching** Feature matching prevents the generator from over-training by requiring it to generate data whose feature map matches the real one.
* **Mini-batch Discrimination** Mini-batch discrimination looks at multiple examples in combination, rather than in isolation, and could potentially help avoid collapse of the generator.
* **Historical Averaging** Historical averaging takes the average of historical parameters into consideration when updating them.
* **One-sided Label Smoothing** Replaces the 0 and 1 targets for a classifier with smoothed values, like .9 or .1, which was recently shown to reduce the vulnerability of neural networks to adversarial examples.
* **Virtual Batch Normalization** In order to avoid the situation where an input example is highly dependent on several other inputs in the same mini-batch, choose a reference batch at the start of training and normalize each example during the rest of the training process. 

CatGAN with feature matching shows better performance in semi-supervised CIFAR-10 classification. Even though mini-batch discrimination enhances the images generated by the generator of GANs, it fails to improve the performance of semi-supervised classification.


[krizhevsky2012imagenet]: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[girshick2014rich]: http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf
[girshick2015fast]: http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
[ren2015faster]: http://papers.nips.cc/paper/5638-analysis-of-variational-bayesian-latent-dirichlet-allocation-weaker-sparsity-than-map.pdf
[goodfellow2014generative]: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
[mirza2014conditional]: http://arxiv.org/pdf/1411.1784
[denton2015deep]: http://papers.nips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf
[radford2015unsupervised]: http://arxiv.org/pdf/1511.06434
[springenberg2015unsupervised]: http://arxiv.org/pdf/1511.06390
[salimans2016improved]: http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf
