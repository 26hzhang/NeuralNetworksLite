# Neural Networks Lite

![Authour](https://img.shields.io/badge/Author-Zhang%20Hao%20(Isaac%20Changhau)-blue.svg) ![](https://img.shields.io/badge/Java-1.8-brightgreen.svg) ![](https://img.shields.io/badge/DeepLearning4J-0.8.0-yellowgreen.svg) ![](https://img.shields.io/badge/ND4J-0.8.0-yellowgreen.svg)

This repository contains three projects:
- [NeuralNetworks4J](/NeuralNetworks4J), which implements several famous neural networks only depend on pure Java (without any third party dependencies).
- [NeuralNetworksND4J](/NeuralNetworksND4J), re-implement the [NeuralNetworks4J](/NeuralNetworks4J) using ND4J library (a scientific library for Java).
- [NeuralNetworksDL4J](/NeuralNetworksDL4J), interesting codes to solve some practical tasks using Deeplearning4j library (a powerful library for Java to tackle deep learning tasks).

Among this three projects, which generally contains several following neural networks algorithms:
* (Multi-Layer) Perceptrons
* Logistic Regression
* Restricted Boltzmann Machines
* Deep Belief Nets
* Denoising Autoencoder
* Stacked Denoising Autoencoder
* Convolutional Neural Networks
* Recurrent Neural Networks (LSTM)

### Requirements
* [ND4J](http://nd4j.org) Library. For [Maven](http://mvnrepository.com/artifact/org.nd4j), import following dependency:
```xml
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```
* [DeepLearning4J](https://deeplearning4j.org) Library. For [Maven](http://mvnrepository.com/search?q=deeplearning4j), import following dependency:
```xml
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-core</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```

### Notes and Information
**Perceptrons**
* Wiki-Link: [Perceptron](https://en.wikipedia.org/wiki/Perceptron).
* Author: [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt).
* Paper: [The Perceptron: A Perceiving and Recognizing Automaton](http://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf).

**Logistic Regression**
* Wiki-Link: [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression).

**Multi-Layer Perceptrons**
* Wiki-Link: [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).
* Author: [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt).
* Paper: [Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms](http://oai.dtic.mil/oai/oai?verb=getRecord&metadataPrefix=html&identifier=AD0256582).

**Restricted Boltzmann Machines**
* Wiki-Link: [Restricted Boltzmann Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine).
* Author: [Paul Smolensky](https://en.wikipedia.org/wiki/Paul_Smolensky).
* Paper: [Information processing in Dynamical Systems: Foundations of Harmony Theory](http://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf).

**Deep Belief Nets**
* Wiki-Link: [Deep Belief Network](https://en.wikipedia.org/wiki/Deep_belief_network).
* Author: [Geoffrey E. Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) et al.
* Paper: [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).

**Denoising Autoencoder**
* Wiki-Link: [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder).
* Author: [Pascal Vincent](http://www.iro.umontreal.ca/~vincentp/).
* Paper: [Extracting and Composing Robust Features with Denoising Autoencoders](http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf), [Deep Learning with Denoising Autoencoders](https://pdfs.semanticscholar.org/bbe9/7e302b1a48345f409c3e935b17ab116455c3.pdf).

**Stacked Denoising Autoencoder**
* Wiki-Link: [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
* Author: [Pascal Vincent](http://www.iro.umontreal.ca/~vincentp/)
* Paper: [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf).

**Convolutional Neural Networks**
* Wiki-Link: [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network).
* Paper: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), [Deep Sparse Rectifier Neural Networks](http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf), [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf), [Yann LeCun](http://yann.lecun.com/exdb/lenet/)'s page.

**Recurrent Neural Networks**
* Wiki-Link: [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network), [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory).
* Usage: [dl4j-rnns](https://deeplearning4j.org/usingrnns).
* Paper: [Bidirectional Recurrent Neural Networks](https://maxwell.ict.griffith.edu.au/spl/publications/papers/ieeesp97_schuster.pdf), [Long Short-Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory), [Jürgen Schmidhuber](http://people.idsia.ch/~juergen/rnn.html)'s page.

**Reference**: [Java Deep Learning Essentials](https://github.com/PacktPublishing/Java-Deep-Learning-Essentials)
