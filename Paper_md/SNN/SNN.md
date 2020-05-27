# TRAINING DEEP SPIKING NEURAL NETWORKS FOR ENERGY-EFFICIENT NEUROMORPHIC COMPUTING

## Abstract

3 different SNN training methodologies:

1. STDP
    a. limited shallow SNNs(4 layers)
2. ANN-to-SNN
    a. leads to high inference latency
3. Spike-based error backpropagation algorithm
    a. reduce the inference latency by up to 8 x

## 1. Introduction

![Training methodologies](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200520175948-snn.png)

## 2. SNN PRELIMINARIES

![core buliding block](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200520172817-snn.png)

1. Object recognition: image pixels-->Poisson spike trains firing
2. Post-neuron: emulated using either Integrate-and-Fire (IF) or Leaky-Integrateand-Fire (LIF) model

## 3. SNN TRAINING APPROACHES AND RESULTS

### 3.1. Spike Timing Dependent Plasticity (STDP)

### 3.2. ANN-SNN Conversion

![ANN-SNN](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200520180549.png)

1. Training ANN without BN and max pooling
2. Transfer the trained weights from ANN to SNN
3. Initialize the firing threshold of IF neuron in every SNN layer

Threshold balancing:

1. Low threshold, too high increases the inference latency
    a. Spike-norm algorithm in layer-wise manner
2. First generate Poisson spike trains and propagate to the first layer of SNN
3. Record the first layer's weighted spike-input and set the threshold to the max
4. The same process is carried out

### 3.3. Spike-Based Error Backpropagation

1. LIF pseudo-derivative
2. Dropout, L2 regularization
