by Yikun Gu

## PISeL: Pipelining DNN Inference for Serverless Computing

![image-20241218150436665](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241218150436665.png)

![image-20241218150642643](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241218150642643.png)

### Introduction

DNN应用程序引导（DNN框架加载和启动、模型初始化、模型下载、反序列化和复制）是整个冷启动过程中的主要因素。随着模型大小的增长，应用程序级引导变得更加严重。

insight： DNN模型通常由多个层组成，请求遵循逐层执行模式。

我们设计了一种流水线模型机制，可以将模型从远程存储下载、模型反序列化和加载以及请求执行进行流水线处理。

### Motivation

#### 冷启动

![image-20241218165820076](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241218165820076.png)

#### 内存占用



![image-20241219075248508](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219075248508.png)

#### Transparency To Frameworks and Platforms

我们的工作通过提出一种框架无关的方法来解决这个问题，该方法可以很容易地与TensorFlow、PyTorch和MXNet等流行的DNN框架集成，而无需对应用程序代码或无服务器平台进行大量修改。通过使用动态库插入和函数级挂钩，我们的解决方案可以在各种无服务器环境中透明地优化DNN推理工作负载。



### System Design

![image-20241219080425786](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219080425786.png)

#### Model Partitioning

PISeL的第一阶段涉及将DNN模型拆分为多个分区，每个分区由一个或多个层组成。对模型进行分区可以实现层的并行下载、加载和执行，从而显著减少冷启动时间和峰值内存使用。

切分粒度：将单层拆分为多个分区在冷启动期间的内存使用或响应时间方面没有任何好处。这是因为，在加载阶段，无论如何分区，都必须将整个层传递给DNN框架。因此，我们的设计确保了每一层在分区内保持完整。

此外，为了通过并行执行和加载实现最佳性能，我们的分区策略仅对分区内的连续层进行分组。这允许以正确的顺序加载层，使DNN框架能够在解决层的依赖关系后立即开始执行层。

challenge：确定连续层到分区的最佳分组。两个主要考虑因素指导着这一决定。首先，创建新分区会引入一个开销，该开销等于与初始化分区相关的请求延迟（RL）。其次，在没有开销的情况下，将峰值内存使用和冷启动响应时间降至最低的理想方案是将每一层放置在自己的分区中。然而，由于管理大量分区带来的开销，这种方法并不实用。为了应对这一挑战，我们为每个分区的大小引入了一个下限。如果前一个分区的大小超过RL和连接带宽（BW）的乘积，则可以避免分区开销。



#### Model Partitioning Solver

![image-20241219083936982](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219083936982.png)

分区下载器并行下载分区，当当前分区的剩余大小等于覆盖大小（RL×BW）时启动下一次下载，最大限度地提高网络带宽利用率。分区加载器和执行器在每个分区内加载和执行层，使推理请求能够在不等待整个模型加载的情况下得到处理。条件锁同步加载和执行过程。

#### Parallelize Model Loading And Execution

为了进一步减少冷启动时间，PISeL并行化了模型加载和请求执行。这是可能的，因为一旦加载了权重，层就可以执行请求，并且执行过程会逐层进行。通过利用这一特性，我们可以在加载剩余层的同时开始执行对加载层的请求。

challenges：

- 只有在该层中的所有权重都已完全加载后，才能在该层执行请求，如果权重尚不可用，则应等待。
- 对于每一层，并行化机制引入的开销必须最小。考虑到复杂DNN模型中的大量层，即使每层的开销很小，也会累积并显著增加响应时间。
- 不同类型的层具有不同数量的参数（例如权重、偏差），有些层可能根本没有任何参数。并行化机制必须考虑到这些差异。

##### condition locking mechanism：

当模型初始化时，所有锁都会关闭。加载层参数时，相应的锁打开，表示层已准备好执行。为了尽量减少开销，PISeL不会对没有任何参数的层应用锁。这种优化减少了锁操作的次数并提高了效率。此外，为了处理不同类型的层及其不同的参数配置，PISeL动态地识别每一层的所有持久参数。只有当层的所有参数都完全加载时，层的锁定条件才会更新。这确保了只有当一个层具有所有必要的可用参数时，它才会被执行。通过使用这种条件锁定机制，PISeL有效地同步了模型加载和请求执行过程。具有加载权重的层可以立即开始执行请求，而具有挂起权重的层则等待其完全加载。这种并行化大大缩短了冷启动时间，因为层的执行可以比传统的顺序方法早得多。

#### Transparency and Compatibility

PISeL设计的一个关键方面是它对DNN作业和无服务器平台的透明度，以及对DNN框架的兼容性。PISeL需要最少的代码更改才能与现有系统集成，使其具有高度的适应性和易于采用。钩子和包装器的使用允许PISeL轻松地整合到各种DNN框架中，如PyTorch、TensorFlow和MXNet。这种透明度和兼容性使更广泛的社区能够从PISeL的优化中受益，而不需要对现有的代码库进行大量修改。通过避免对框架进行深入更改，PISeL保持了与不同版本的DNN框架的兼容性，确保了顺利的集成过程。此外，PISeL的设计与平台无关，这意味着它可以很容易地部署在各种无服务器计算平台上，而不需要任何特定于平台的修改。这种灵活性允许用户在不同的无服务器提供商中利用PISeL的优势，进一步增强其可用性和采用潜力。

### Evaluation

#### Setup

我们在CPU和GPU平台上评估PISeL。CPU设置包括一台配备2.80GHz 24核AMD 7402P CPU、128GB ECC内存（8x16GB 3200MT/s RDIMM）和1.6TB NVMe SSD（PCIe v4.0）的服务器。GPU设置包括一台服务器，配备两个3.20GHz的Intel Xeon E5-2667 8核CPU、128GB ECC内存、两个960GB 6G SATA SSD和一个NVIDIA 16GB Tesla V100 SMX2 GPU。这两种设置都连接到具有10Gb/s吞吐量的多节点、多存储MinIO对象存储。

#### Workloads

DNN models used include ResNet50 , VGG19 , RegNet , GPT2 , LaBSE , GPT2-XL , Wav2Vec2 , Whisper-M , and Whisper-L . We use standard configurations for each model.

#### Metrics

我们测量了延迟和峰值内存使用情况，报告了30次运行的平均值。通过将禁用PISeL的响应时间或峰值内存使用量除以启用PISeL时的相同度量来计算提升。

#### Baselines

我们的基线是“无优化”和“逐层”管道。“无优化”按顺序执行阶段（下载、反序列化、加载、执行），称为“无管道”。“逐层”逐层下载模型，反序列化和加载遵循相同的模式。



#### Latency in Cold-Start

![image-20241219122557192](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219122557192.png)

![image-20241219122622642](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219122622642.png)

1）随着模型大小的增加，加速率也会增加，因为DNN模型引导时间在总冷启动时间中所占的百分比越来越大。这一趋势如§2的图1所示。

2） 随着批量大小的增加，加速率先增加后减小。这是由于随着批处理大小的增加，执行延迟也在增加。一旦执行延迟主导了整个冷启动时间，下载、反序列化和加载以及执行的阶段延迟在任何分区模式中都无法很好地重叠，从而影响了流水线效率。

3） 对于小型模型，由于DNN模型引导在整体冷启动中所占比例较小，PISeL不会增加显著的好处或开销，在不使用管道的情况下，其性能与基线相似。

4） PISeL显示了CPU和GPU平台上的性能提升。然而，图5中TensorFlow在GPU上的加速通常低于图7所示的PyTorch在GPU上。这是因为TensorFlow上的模型没有得到很好的优化，导致性能不佳。然而，PISeL在Regnet和GPT2-XL等大型型号上仍然可以实现高达1.32倍的加速。

#### Peak Memory Usage

![image-20241219122903703](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219122903703.png)

对于PyTorch，大型模型的内存节省最为明显，PISeL将GPT2-XL、Regnet和Whisper large的峰值使用率降低了2倍以上。这是通过将模型参数分成不同的组，然后在不同的时隙单独加载每个组来实现的。这样，与复制整个模型参数相比，它可以显著降低加载阶段的内存消耗。例如，用vanilla PyTorch加载GPT2-XL需要超过18GB的GPU内存，而PISeL将其减少到8GB以下。内存节省较小，但对TensorFlow来说仍然很重要，范围在1.22−1.42×之间。较低的节省是由于TensorFlow中现有的优化将模型加载拆分为块。然而，PISeL仍然将GPT2-XL的峰值使用量减少了4GB，从13GB减少到9GB，这是一个显著的减少，特别是在内存有限的GPU上部署时。对于像ResNet和Wav2Vec2这样的小型模型，内存开销并不那么突出，因为模型本身要小得多。然而，PISeL始终比基线使用更少的内存。这些结果突出了PISeL内存管理的重要性。通过启用增量加载，PISeL大大减少了内存占用，允许在给定的GPU上部署更大的模型，并实现更密集的多租户。

#### Pipelined Model Transmission and Loading

![image-20241219122950633](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219122950633.png)

图10和图11显示了客户端从发出推理任务到完成推理的总时间，其中包括容器创建、运行时和库加载以及DNN模型引导的延迟。逐层管道在层的粒度上与下载、反序列化和加载以及计算重叠。但是，由于它在每一层都有下载和加载开销以及同步开销，因此它在小型和大型模型的Tensorflow和Pytorch框架上的性能都较差。对于逐层和PISeL流水线机制，随着批量的进一步增加，我们没有看到持续的加速增长。

这是因为随着更大批量，计算量预计会相应增加，如果一个阶段的时间很长，这会打破跨阶段的重叠。然后，管道效率受到影响。

#### Partitioning Time

我们测量了PISeL的最优分组算法为PyTorch和TensorFlow划分模型所花费的时间。图12（a）显示了PyTorch的结果，而图12（b）显示了TensorFlow的结果。对于PyTorch模型，分区时间从VGG19的3.5毫秒到Whisper-L的167毫秒不等，随着模型大小和复杂性的增加而增加。即使是最大的模型（Whisper-L）也能在200毫秒内进行分区，与加载和推理时间相比可以忽略不计。TensorFlow模型显示出类似的趋势，由于测试的模型较少，时间从VGG19的1.5毫秒到GPT2-XL的40毫秒不等。这些时间对推理延迟的开销最小。PISeL的高效算法通过基于计算和传输时间修剪次优分区以及违反内存约束的分区来实现这些低时间，从而允许快速优化分区。

![image-20241219123220475](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241219123220475.png)
