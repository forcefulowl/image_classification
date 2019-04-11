
# Image_classification

## Introduction

Convolutional neural networks ahve become famous in cmoputer vision ever since *AlexNet* popularized deep convolutional neural networks by winning IamgeNet Challenge: ILSVRC 2012. The general trend has been to make deeper and more complicated networks in roder to achieve higher accuracy. However, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and spped. In many real world applications such as robotics, self-driving, the recognition tasks need to be carried out in a timely fashion on a computationally limited platform. That's inspire me to build some effient convolutional neural networks which can be used on Mobile/portable device. 

## Data/Preprocessing Data

The input data are dermoscopic lesion images in JPEG format.

The training data consists of 10015 images.

```
AKIEC: 327
BCC: 514
BKL: 1099
DF: 115
MEL: 1113
NV: 6705
VASC: 142
```

The format of raw data is as follows:

<img src='/img/raw_data.png'>

And the format of label is as follows:

<img src='/img/raw_label.png'>

Directly loading all of the data into memory.

```
def read_img(img_name):
    im = Image.open(img_name).convert('RGB')
    data = np.array(im)
    return data

images = []

for fn in os.listdir('C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input'):
    if fn.endswith('.jpg'):
        fd = os.path.join('C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input', fn)
        images.append(read_img(fd))
```

That is so memory consuming, even the most state-of-the art configuration won't have enough memory space to process the data the way I used to do it. Meanwhile, the number of training data is not large enough, Data Augumentation is the next step to achieve.

Firstly, chaning the format of the raw data using `reformat_data.py`.

<img src='/img/new_data.png'>

Then doing data augumentation.

```
data_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    validation_split=0.3)
```

`horizontal_flip`: Randomly flip inputs horizontally.
`zoom_range`: Range for random zoom.
`width_shift_range`: fraction of total width.
`height_shift_range`: fraction of total height.
`rotation_range`: Degree range for random rotations.
`validation_split`: Split the dataset into 70% train and 30% val.

Then achieve ImageGenerator:

```
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)
```


## Model/ Tricks to imporve performance on mobile device


### Depthwise Separable Convolution

Depthwise Separable Convolution is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution called a pointwise convolution. The depthwise convolution applies a single filter to each input channel, the pointwise convolution then applies a ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size. 

A standard convolutional layer takes as input a ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D%20%5Ctimes%20D_%7BF%7D%20%5Ctimes%20M) feature map F and produces a ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D%20%5Ctimes%20D_%7BF%7D%20%5Ctimes%20N) feature map G where ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D) is a spatial width and height of a square input feature map, M is the number of input channels, ![](https://latex.codecogs.com/gif.latex?D_%7BG%7D) is the spatial width and height of a square output feature map and N is the number of output channel.

The standard convolutional layer is parameterized by convolution kernel K of size ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ctimes%20D_%7BK%7D%20%5Ctimes%20M%20%5Ctimes%20N) where ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D) is the spatial dimension of the kernel assumed to be square and M is number of input channels and N is the number of output channels as defined previously. 

Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. Using depthwise convolutions to apply a single filter per input channel (input depth). Pointwise convolution, a simple ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution, is then used to create a linear combination of the output of the depthwise layer.

Depthwise convolution with one filter per input channel (input depth) can be written as:


![](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%2Cj%7D%20K_%7Bi%2Cj%2Cm%7D%20%5Ccdot%20F_%7Bk&plus;i-1%2Cl&plus;j-1%2Cm%7D)


where K is the depthwise convolutional kernel of size ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ctimes%20D_%7BK%7D%20%5Ctimes%20M) where the ![](https://latex.codecogs.com/gif.latex?m_%7Bth%7D) filter in K is applied to the ![](https://latex.codecogs.com/gif.latex?m_%7Bth%7D) channel in F to produce the ![](https://latex.codecogs.com/gif.latex?m_%7Bth%7D) channel of the filtered output feature map G.


<img src = '/img/depthwise separable convolution.png'>


Depthwise convolution has a computational cost of


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)


Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution is needed in order to generate these new features.

The combination of depthwise convolution and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) (pointwise) convolution is called depthwise separable convolution which was originally introduced in.

Depthwise separable convolutions cost:


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%20&plus;%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)


which is the sum of the depthwise and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) pointwise convolutions.

By expressing convolution as two step process of filtering and combining, there's a reduction in computation of


![](https://latex.codecogs.com/gif.latex?%5Cfrac%7BD_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%20&plus;%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%7D%7BD_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20&plus;%20%5Cfrac%7B1%7D%7BD%5E2_%7BK%7D%7D)



| Type/Strike |  Filter shape | Input Size |
| ------ | ------ | ------ | 
| Conv2D/s=2 | 3 * 3 * 32 | 224 * 224 * 3 |
| DWConv2D/s=1 | 3 * 3 * 32dw | 112 * 112 * 32 |
| Conv2D/s=1 | 1 * 1 * 32 * 64 | 112 * 112 * 32 |
| DWConv2D/s=2 | 3 * 3 * 64dw | 112 * 112 * 64 |
| Conv2D/s=1 | 1 * 1 * 64 * 128 | 56 * 56 * 64 |
| DWConv2D/s=1 | 3 * 3 * 128dw | 56 * 56 * 128 |
| Conv2D/s=1 | 1 * 1 * 128 * 128 | 56 * 56 * 128 |
| DWConv2D/s=2 | 3 * 3 * 128dw | 56 * 56 * 128 |
| Conv2D/s=1 | 1 * 1 * 128 * 256 | 28 * 28 * 256 |
| DWConv2D/s=1 | 3 * 3 * 256dw | 28 * 28 * 256 |
| Conv2D/s=1 | 1 * 1 * 256 * 256 | 28 * 28 * 256 |
| DWConv2D/s=2 | 3 * 3 * 256dw | 28 * 28 * 256 |
| Conv2D/s=1 | 1 * 1 * 256 * 512 | 14 * 14 * 256 |
| 5 * DWConv2D, Conv2D/s=1 | 3 * 3 * 512dw, 1 * 1 * 512 * 512 | 14 * 14 * 512 |
| DWConv2D/s=2 | 3 * 3 * 512dw | 14 * 14 * 512 |
| Conv2D/s=1 | 1 * 1 * 512 * 1024 | 7 * 7 * 512 |
| DWConv2D/s=1 | 3 * 3 * 1024dw | 7 * 7 * 1024 |
| Conv2D/s=1 | 1 * 1 * 1024 * 1024 | 7 * 7 * 1024 |
| Pooling/FC |  |  |


#### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=32, lr=1, epochs=100 | 2:02:35.984546 | loss: 0.0496 - acc: 0.9844 - val_loss: 0.9146 - val_acc: 0.8437 |
| batch_size=32, lr=0.1, epochs=100 | 2:02:24.674883 | loss: 0.0167 - acc: 0.9938 - val_loss: 0.8037 - val_acc: 0.8592 |
| batch_size=32, lr=0.01, epochs=100 | 2:00:44.432700 | loss: 0.1524 - acc: 0.9481 - val_loss: 0.5204 - val_acc: 0.8420 |
| batch_size=32, lr=0.001, epochs=100 | 1:59:59.366265 | loss: 0.5579 - acc: 0.8000 - val_loss: 0.6101 - val_acc: 0.7804 |
| batch_size=16, lr=0.1, epochs=100 | 2:11:34.221687 | loss: 0.0326 - acc: 0.9896 - val_loss: 0.8072 - val_acc: 0.8539 |
| batch_size=16, lr=0.01, epochs=100 | 2:13:38.293478 | loss: 0.1412 - acc: 0.9524 - val_loss: 0.5066 - val_acc: 0.8372 |
| batch_size=8, lr=0.1, epochs=100 | 2:29:13.814567 | loss: 0.0488 - acc: 0.9836 - val_loss: 0.8756 - val_acc: 0.8473 |



#### Inverted Residuals and Linear Bottlenecks

**Deep Residual Learning for Image Recognition**

Deep convolutional neural networks have led to a series of breakthroughs for image classification. Deep networks naturally integrate low/mid/high-level features and classifiers in an end-to-end multilayer fashion, and the 'levels' of features can be enriched by the number of stacked layers(depth). Evidence reveals that network depth is of crucial importance.

Driven by the significance of depth, a question arises: *Is learning better networks as easy as stacking more layers?*  An obstacle to answering this question was the notorious problem of vanishing/exploding gradients, which hamper convergence from the beginning. When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing ,accuracy gets saturated ( which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.

<img src='/img/gradient_vanishing.png'>

Someone address the degradation problem by introducing a *deep residual learning* framework. In stead of hoping each few stacked layers directly fit a desired underlying mapping, we can let these layers fit a residual mapping. 

<img src='/img/residual block.png'>

Formally, denoting the desired underlying mapping as ![](https://latex.codecogs.com/gif.latex?H%28x%29), it let the stacked nonlinear layers fit another mapping of ![](https://latex.codecogs.com/gif.latex?F%28X%29%3A%3DH%28x%29%20-%20x). The original mapping is recast into ![](https://latex.codecogs.com/gif.latex?F%28x%29%20&plus;%20x). The formulation of ![](https://latex.codecogs.com/gif.latex?F%28x%29%20&plus;%20x) can be realized by feedforward neural networks with 'shortcut connections' are those skipping one or more layers, the shortcut connections simply perform *identity* mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcut connections add neither extra parameter nor computational complexity.

### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=8, lr=1, epochs=100 | 2:47:20.486248 |loss: 0.3449 - acc: 0.8718 - val_loss: 1.1156 - val_acc: 0.7741|
| batch_size=8, lr=0.01, epochs=200 | 5:41:32.741626 |loss: 0.0190 - acc: 0.9937 - val_loss: 1.1468 - val_acc: 0.8460|


**Deeper Bottleneck Architectures**

<img src='/img/bottleneck.png'>

For each residual function ![](https://latex.codecogs.com/gif.latex?F), using a stack of 3 layers instead of 2. The three layers are ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201), ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203), and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolutions, where the ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) layers are responsible for reducing and then increasing(restoring) dimensions, leaving the ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) layer a bottleneck with smaller input\output dimensions.

**Linear Bottlenecks**

Comparing of Depthwise Separable Convolution and Linear Bottleneck.

<img src='/img/comparing of mobilenet v1_v2.png'>

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Using linear layers is crucial as it prevents non-linearities from destroying too much information.

<img src='/img/linear bottleneck.png'>

Examples of ReLU transformations of low-dimensional manifolds embedded in higher-dimensional spaces. In these examples the initial spiral is embedded into an n-dimensional space using random matrix T followed by ReLU, and then projected back to the 2D space using $T^{-1}$. In examples above n = 2,3 result in information loss where certain points of the manifold collapse into each other, while for n=15 to 30 the transformation is highly non-convex.

**Inverted residuals**

The inverted design is considerably more memory efficient.

<img src='/img/inverted block.png'>

Comparing of bottleneck and inverted residuals.

<img src='/img/comparing of bottleneck.png'>


#### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=32, lr=1, epochs=100 | 2:01:26.619140 |loss: 0.0795 - acc: 0.9733 - val_loss: 1.6232 - val_acc: 0.7831|
| batch_size=32, lr=0.1, epochs=100 | 2:01:06.484069 |loss: 0.0183 - acc: 0.9941 - val_loss: 1.1009 - val_acc: 0.8404|
| batch_size=32, lr=0.01, epochs=100 | 2:01:09.871235 |loss: 0.1427 - acc: 0.9521 - val_loss: 0.5747 - val_acc: 0.8397|


#### Channel shuffle for Group Convolution

Modern convolutional neural networks usually consist of repeated building blocks with the same structure, such as *Xception* and *ResNeXt* introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, both designs do not fully take the ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolutions into account, which require considerable complexity. For example, in ResNeXt only ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) layers are equipped with group convolutions. As a result, for each residual unit in ResNeXt the pointwise convolutions occupy 93.4% multiplication-adds( cardinality = 32 as suggested in). In tiny networks, expensive pointwise convolutions result in limited number of channels to meet the complexity constraint, which might significantly damage the accuracy.

To address the issue, a straightforward solution is to apply channel sparse connections, for example group convolutions, also on ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) layers.By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels. It is clear that outputs from a certain group only relate to the inputs within the group. This property blocks information flow between channel groups and weakens representation

If we allow group convolution to obtain input data from different groups , the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups. 

<img src='/img/channel_shuffle.png'>


|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=16, lr=0.1, epochs=200 | 4:40:04.491828 |loss: 0.2473 - acc: 0.9078 - val_loss: 0.8477 - val_acc: 0.7829|


#### ShuffleNet Unit

Starting from the design principle of bottleneck unit. In its residual brach, for the $3 \times 3$ layers, applying a computational economical $3 \times 3$ depthwise convolution on the bottleneck feature map. Then, replacing the first $1 \times 1$ layer with pointwise group convolution followed by a channel shuffle operation. The purpose of the second pointwise group convolution is to recover the channel dimension to match the shortcut path.

<img src='/img/shufflenet_unit.png'>


## TODO

https://arxiv.org/pdf/1807.11164.pdf
