​                          **YOLOv4**

**《YOLOv4: Optimal Speed and Accuracy of Object Detection》**

背景。贡献。解决什么问题。有待改进之处。

**如何构建这个模型！**

**一、引言**

​  1、V3 回顾

​      backbone：darknet53

​      anchor boxes

​      batch normalization

​      multi-scale  (test544)

​      取消了pooling层，改用卷积核步长进行降采样

​      logistics代替softmax

​  2、

​      V4 中，作者做了很多实验，把近几年的一些方法加入yolo中，最终取得了   效果和速度的提升。通过了解yolov4，我们可以很好的知道近几年有哪些方法    被提出来，整篇论文更像是一篇综述，但是和绝大多数综述不同的是，作者是    有做实验提炼出一个优秀的模型的！

​      *Weighted-Residual-Connections*

​      *Cross-Stage-Partial-connections*

​      Cross mini-Batch Normalization

​      *Self-adversarial-training*

​      *Mish activation*

​      *Mosaic data augmentation*

​      *DropBlock regularization*

​      *CIoU loss*



**二、Bag of freebies**

​  改变培训策略，或者只会增加培训成本的方法，对测试不影响。

​  **数据扩充：**

​      1、光度畸变：调整图像的亮度、对比度、色调、饱和度和噪声

​      2、几何畸变：加入随机缩放、剪切、翻转和反旋转

​  **模拟对象遮挡：**

​      1、random erase，CutOut：可以随机选择图像中的矩形区域，并填充一 个随机的或互补的零值

​      2、hide-and-seek、grid mask：随机或均匀地选择图像中的多个矩形区 域，并将其全部替换为0

​  **feature map：**

​      DropOut、DropConnect和DropBlock。

​  **结合多幅图像进行数据扩充：**

​      MixUp、CutMix

​  **Style Transfer GAN**

​  **解决类别不平衡：**

​      hard negative example mining (只适用两阶段)

​      online hard example mining (只适用两阶段)

​      focal loss

​  **label smoothing**

​  **bbox：**

​      1、IoU_loss

​      2、GIoU_loss

​      3、DIoU_loss

​      4、CIoU_loss

​  **YOLOv4 - use：**

​      *CutMix and Mosaic data augmentation*、DropBlock regularization、   Class label smoothing、CIoU-loss、*CmBN*、*Self-Adversarial Training*、     *Eliminate grid sensitivity*、Using multiple anchors for a single ground        truth、Cosine annealing scheduler、Optimal hyperparameters、Random      training shapes。



**三、Bag of specials**

​  只会增加少量推理成本但却能显著提高对象检测精度的plugin modules和post-processing methods

​  **enhance receptive field**：SPP，ASPP，RFB

​  **attention module:**

​      1、Squeeze-and-Excitation (SE)：可以改善resnet50在分类任务上提高    1%精度，但是会增加GPU推理时间10%。

​      2、Spatial Attention Module (SAM)：可以改善resnet50在分类任务上提   高0.5%精度，并且不增加GPU推理时间。

​  **feature integration：**

​      早期使用skip connection、hyper-column。随着FPN等多尺度方法的流  行，提出了许多融合不同特征金字塔的轻量级模型。SFAM、ASFF、BiFPN。   SFAM的主要思想是利用SE模块对多尺度拼接的特征图进行信道级配重权。    ASFF使用softmax作为点向水平重加权，然后添加不同尺度的特征映射。     BiFPN提出了多输入加权剩余连接来执行按比例加权的水平重加权，然后加入不   同比例的特征映射。

​  **activation function：**

​      ReLU解决了tanh和sigmoid的梯度消失问题。                             LReLU ， PReLU ， ReLU6 ，SELU， Swish ， hard-Swish ， Mish 其中   Swish和Mish都是连续可微的函数。

​  **post-processing method**

​      nms：c·p

​      soft-nms：解决对象的遮挡问题

​      DIoU nms：将中心点分布信息添加到BBox筛选过程中

​  **YOLOv4 - use：**

​      Mish activation、CSP、MiWRC、SPP-block、SAM、PAN、DIoU-NMS

​

**四、Selection of architecture**

​  在ILSVRC2012 (ImageNet)数据集上的分类任务，CSPResNext50要比CSPDarknet53好得多。然而，在COCO数据集上的检测任务，CSP+Darknet53比CSPResNext50更好。

​  backbone：CSP+Darknet53

​  additional module：SPP

​  neck：PANet

​  head：YOLOv3 (anchor based)



**五、Additional improvements**

​  为了使检测器更适合于单GPU上的训练，做了如下补充设计和改进:

​  1、引入了一种新的数据增强方法Mosaic和自对抗训练(SAT)

​  2、在应用遗传算法的同时选择最优超参数

​  3、修改了一些现有的方法，如：SAM，PAN，CmBN



**六、细节**
