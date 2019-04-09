<head>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
  inlineMath: [['$','$'], ['\\(','\\)']],
  processEscapes: true
  }
});
</script>
<!--latex数学显示公式-->
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
</head>

# image_classification

#### Preprocessing Dataset



#### Depthwise Separable Convolution

Depthwise Separable Convolution is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a $1 \times 1$ convolution called a pointwise convolution. The depthwise convolution applies a single filter to each input channel, the pointwise convolution then applies a $1 \times 1$ convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size. 

A standard convolutional layer takes as input a $D_{F} \times D_{F} \times M$ feature map F and produces a $D_{F} \times D_{F} \times N$ feature map G where $D_{F}$ is a spatial width and height of a square input feature map, M is the number of input channels, $D_{G}$ is the spatial width and height of a square output feature map and N is the number of output channel.

The standard convolutional layer is parameterized by convolution kernel K of size $D_{K} \times D_{K} \times M \times N$ where $D_{K}$ is the spatial dimension of the kernel assumed to be square and M is number of input channels and N is the number of output channels as defined previously. 

Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. Using depthwise convolutions to apply a single filter per input channel (input depth). Pointwise convolution, a simple $1 \times 1$ convolution, is then used to create a linear combination of the output of the depthwise layer.

Depthwise convolution with one filter per input channel (input depth) can be written as:
$$
\sum_{i,j} K_{i,j,m} \cdot F_{k+i-1,l+j-1,m}
$$
where K is the depthwise convolutional kernel of size $D_{K} \times D_{K} \times M$ where the $m_{th}$ filter in K is applied to the $m_{th}$ channel in F to produce the $m_{th}$ channel of the filtered output feature map G.

![](C:\Users\gavin\Desktop\img_classification\depthwise separable convolution.png)

Depthwise convolution has a computational cost of
$$
D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}
$$
Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via $1 \times 1$ convolution is needed in order to generate these new features.

The combination of depthwise convolution and $1 \times 1$ (pointwise) convolution is called depthwise separable convolution which was originally introduced in.

Depthwise separable convolutions cost:
$$
D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F} + M \cdot N \cdot D_{F} \cdot D_{F}
$$
which is the sum of the depthwise and $1 \times 1$ pointwise convolutions.

By expressing convolution as two step process of filtering and combining, there's a reduction in computation of
$$
\frac{D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F} + M \cdot N \cdot D_{F} \cdot D_{F}}{D_{K} \cdot D_{K} \cdot M \cdot N \cdot D_{F} \cdot D_{F}}
= \frac{1}{N} + \frac{1}{D^2_{K}}
$$


#### Inverted Residuals and Linear Bottlenecks

**Deep Residual Learning for Image Recognition**

Deep convolutional neural networks have led to a series of breakthroughs for image classification. Deep networks naturally integrate low/mid/high-level features and classifiers in an end-to-end multilayer fashion, and the 'levels' of features can be enriched by the number of stacked layers(depth). Evidence reveals that network depth is of crucial importance.

Driven by the significance of depth, a question arises: *Is learning better networks as easy as stacking more layers?*  An obstacle to answering this question was the notorious problem of vanishing/exploding gradients, which hamper convergence from the beginning. When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing ,accuracy gets saturated ( which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.

![](C:\Users\gavin\Desktop\img_classification\gradient_vanishing.png)

Someone address the degradation problem by introducing a *deep residual learning* framework. In stead of hoping each few stacked layers directly fit a desired underlying mapping, we can let these layers fit a residual mapping. 

![](C:\Users\gavin\Desktop\img_classification\residual block.png)

Formally, denoting the desired underlying mapping as $H(x)$, it let the stacked nonlinear layers fit another mapping of $F(X):=H(x) - x$. The original mapping is recast into $F(x) + x$. The formulation of $F(x) + x$ can be realized by feedforward neural networks with 'shortcut connections' are those skipping one or more layers, the shortcut connections simply perform *identity* mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcut connections add neither extra parameter nor computational complexity.

**Deeper Bottleneck Architectures**

![](C:\Users\gavin\Desktop\img_classification\bottleneck.png)

For each residual function $F$, using a stack of 3 layers instead of 2. The three layers are $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolutions, where the $1 \times 1$ layers are responsible for reducing and then increasing(restoring) dimensions, leaving the $3 \times 3$ layer a bottleneck with smaller input\output dimensions.

**Linear Bottlenecks**

Comparing of Depthwise Separable Convolution and Linear Bottleneck.

![](C:\Users\gavin\Desktop\img_classification\comparing of mobilenet v1_v2.png)

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Using linear layers is crucial as it prevents non-linearities from destroying too much information.

![](C:\Users\gavin\Desktop\img_classification\linear bottleneck.png)

Examples of ReLU transformations of low-dimensional manifolds embedded in higher-dimensional spaces. In these examples the initial spiral is embedded into an n-dimensional space using random matrix T followed by ReLU, and then projected back to the 2D space using $T^{-1}$. In examples above n = 2,3 result in information loss where certain points of the manifold collapse into each other, while for n=15 to 30 the transformation is highly non-convex.

**Inverted residuals**

The inverted design is considerably more memory efficient.

![](C:\Users\gavin\Desktop\img_classification\inverted block.png)

Comparing of bottleneck and inverted residuals.

![](C:\Users\gavin\Desktop\img_classification\comparing of bottleneck.png)

#### Channel shuffle for Group Convolution

Modern convolutional neural networks usually consist of repeated building blocks with the same structure, such as *Xception* and *ResNeXt* introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, both designs do not fully take the $1 \times 1$ convolutions into account, which require considerable complexity. For example, in ResNeXt only $3 \times 3$ layers are equipped with group convolutions. As a result, for each residual unit in ResNeXt the pointwise convolutions occupy 93.4% multiplication-adds( cardinality = 32 as suggested in). In tiny networks, expensive pointwise convolutions result in limited number of channels to meet the complexity constraint, which might significantly damage the accuracy.

To address the issue, a straightforward solution is to apply channel sparse connections, for example group convolutions, also on $1 \times 1$ layers.By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels. It is clear that outputs from a certain group only relate to the inputs within the group. This property blocks information flow between channel groups and weakens representation

If we allow group convolution to obtain input data from different groups , the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups. 

![](C:\Users\gavin\Desktop\img_classification\channel_shuffle.png)

