By Yikun Gu

### StreamBox: A Lightweight GPU SandBox for Serverless Inference Workflow

#### [StreamBox: A Lightweight GPU SandBox for Serverless Inference Workflow](https://www.usenix.org/conference/atc24/presentation/wu-hao)

- 该论文是2024年发表在 ATC 上的论文，其目标是优化 serverless 平台上不同 function 共享 GPU runtime。
- 现有 serverless inference systems 利用 Nvidia Multi-Process Service (MPS)，每个函数有单独的 GPU runtime，造成严重的内存占用（90%的数据冗余）、冷启动时延（超过5s）和函数间通信时重复的数据传输。
- GPU stream 通常由现代 GPU 库（例如 CUDA 和 ROCm）提供，以在 GPU runtime 内实现并发内核执行并共享地址空间，类似于进程中的线程。
- 对实际工作负载的评估表明，与最先进的服务器无感知推理系统相比，StreamBox 将 GPU 内存占用减少了 82%，吞吐量提高了 6.7倍。

#### 论文作者

- Hao Wu, Yue Yu, Junxiao Deng, Ziyue Cheng: Huazhong University of Science and Technology, China
- [Shadi Ibrahim](https://people.rennes.inria.fr/Shadi.Ibrahim/): Inria, Univ. Rennes, CNRS, IRISA, France。 Myriads团队中的Inria研究科学家。目前的研究兴趣是云和雾/边缘计算、可扩展的大数据管理、数据密集型计算、高性能计算、虚拟化技术以及文件和存储系统。
- Song Wu, Hao Fan, Hai Jin: Jinyinhu Laboratory, China
- [Hai Jin](https://www.linkedin.com/in/jinhust/): 他是IEEE和CCF的会员，也是ACM的终身会员。主要研究方向为计算机体系结构、虚拟化技术、集群计算和云计算、点对点计算、网络存储和网络安全。



### 论文动机

#### 问题引入

- 高冗余和过多的内存占用：使用单独的 GPU runtime 隔离函数会导致GPU内存中90%以上的数据冗余（Fig.2）。

<img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001105500784.png" alt="image-20241001105500784"  />



- 不可接受的冷启动开销：GPU运行时冷启动延迟超过5s，常用的预热方法在GPU上无效（Fig.3）。

<img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001105425715.png" alt="image-20241001105425715"  />



- 通信效率低下。由于GPU runtime之间的隔离，函数之间的通信会受到CPU和GPU之间冗余数据复制的影响（Fig.4）。

<img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001110120595.png"  />

- 同一推理工作流的函数不需要很强的隔离，特别是当工作量增加时可以扩展的相同函数。因此，使用stream作为推理工作流的沙盒是一个理想的选择。现代GPU库（例如CUDA和ROCm）通常提供stream以并发执行内核并在GPU runtime共享地址空间。另一方面，stream可以通过纯软件方法对GPU进行分区，例如Elastic Kernel，而不是硬件支持的GPU分区方法（MPS或MIG）。这些方法将内核动态映射到GPU计算单元，而无需重新启动runtime，简化了函数的资源分配。



### SandBox

#### SandBox整体架构

- Auto-scaling内存池，提供细粒度GPU内存管理，并根据自动缩放函数的确切内存使用情况进行弹性缩放。
- 用户透明的通信框架，为开发人员提供统一的通信API来组成函数工作流，并利用弹性通信存储来实现GPU内通信，同时避免与正在运行的函数的内存竞争。
- 细粒度PCIe带宽共享，实现并发函数的高效数据传输。

<img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001111501932.png"  />

#### Auto-scaling内存池

- 细粒度内存管理(Fine-grained Memory Management)：
  - 延迟分配(Lazy allocation)：StreamBox hooks 来自每个函数的内存分配调用（即`cudaMemAlloc()`），并将变量名注册在映射表中。映射表记录（并保持）变量和物理地址之间的映射，其中变量的地址在被访问之前保持为空。当第一次访问变量时，会从GPU内存池中分配一个物理地址；减少了空闲变量对内存的占用。
  - 即时回收(Eager recycling)：在StreamBox中，进行离线预运行推理任务，以获得每个变量将被访问多少次(预期频率)。使用Mapping表记录在一系列内核执行期间每个变量的访问计数(频率)。当变量的访问次数达到预期频率时，可以将该变量标记为可回收的。因此，Eager recycling可以减少无用变量对内存的占用。
- 弹性缩放(Resilient Scaling)：
  - 内存需求分析(Memory demand profiling)：离线地预运行每个模型，以记录其在执行期间的确切内存使用情况，并使用该数据分配以后执行所需的“确切”资源。
  - 实时内存池扩展(Real-time memory pool scaling.)：以固定的间隔定期调整内存池大小，根据离线分析估计未来间隔内的最大内存使用情况。根据下一个时间间隔内的最大内存使用量与当前内存池大小之间的差异调整内存池的大小，这种周期性方法不仅可以准确地感知函数的内存需求，而且可以避免频繁的内存分配和回收。

#### 用户透明的通信框架

- 统一通信框架(Unified Communication Framework)：
  - 通信API(Easy-to-use communication API)：为每个中间数据分配一个全局唯一的索引，然后将其传递给后续函数。
    <img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001130445866.png"  />
  - GPU内部通信优化(Optimizing intra-GPU communication)：在GPU中维护了一个共享通信存储来缓存中间数据。当同一GPU上的后续函数想要访问数据时，它可以直接从通信存储中获取物理地址，而不必像CUDA IPC方法那样获取和打开数据处理程序。
- 弹性通信存储(Elastic Communication Store)：
  - 内存压力感知(Memory pressure awareness)：当内存压力增加时，通信存储会触发中间数据的早期移动。此外，只使用每个GPU中的一部分空闲内存作为通信存储，这样就可以处理突发请求。
  - 自适应GPU间移动(Adaptive inter-GPU movement)：当当前GPU的空闲GPU内存不足时，首先尝试将中间数据移动到相邻的GPU，如果节点中的所有GPU都不足，那么将数据移动到CPU内存中。

#### 细粒度PCIe带宽共享

Stream 只使用PCIe带宽，因为GPU驱动程序中每个GPU runtime只有一个IO引擎。因此，第一个到达的stream垄断了带宽，直到所有数据传输完成，而其他stream的执行由于这种串行数据传输而延迟。将函数的数据划分为更小的数据块。

- 挑战：
  - (1)吞吐量有限。以数据块粒度传输会导致额外的系统开销和带宽浪费;
  - (2)高时延。GPU和CPU之间的数据传输依赖于固定内存，这导致了显著的分配开销;
  - (3)等待时间长。如何允许新到达的请求立即声明它们的PCIe带宽份额;
  - (4)同步无效。由于原始数据已被分割成数据块并由IO进程传输，因此用户程序中的原始同步无效。
- 数据块(Data Block)：
  - 数据块大小(Right-size of data blocks)：根据经验发现，只有当数据块大小超过2MB时，传输带宽才会接近峰值带宽。因此，选择2MB作为StreamBox中的数据块大小。
  - 批处理抢占式传输(Preemptive transfer using batches)：CPU可以异步处理所有传输请求，但会阻塞新到达的请求传输，因此使用batch的方式按批次传输数据块。
  - 共享固定内存缓冲区(Shared pinned memory buffer)：Stream通过主机上的固定内存将数据块传输到GPU内存。然而，分配固定内存会产生巨大的开销（200MB为200ms）。因此，让函数共享固定内存中的一个缓冲区，称为传输缓冲区。
- 数据块同步(Data Block Synchronization)：对于具有固定执行流程的DNN推理，计算过程中的数据依赖关系可以通过离线代码分析获得。因此，在将数据划分为块之后，记录每个块的同步标志。只有每一层的最后一个数据块需要同步。

### 实验

#### 测试平台

实验在GPU服务器上进行，该服务器由两个Intel Xeon（R）Gold 5117 CPU（共28个内核）、128GB DRAM和四个Nvidia V100 GPU（80个CU和16GB内存）组成。软件环境包括PyTorch-1.3.0、TensorFlow-2.12和CUDA-10.1。

#### Workloads

![image-20241025154705330](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241025154705330.png)

- Image Processing：工作流程包括面部识别和图像增强。

- Traffic：工作流程使用一个可以识别车辆和人员的物体检测模型。然后，它对所有相关图像进行后续分析，以进行车辆和面部识别，以及车牌提取。

- Social Media：工作流程将计算机视觉模型与语言模型相结合，根据文本和链接的图像对帖子进行翻译和分类。

- production trace of Azure Function。trace包含7天的请求统计数据，包括每日和每周模式。生产跟踪中有三种典型的请求到达模式，包括零星、周期性和突发性。

#### Baselines

- INFless：INFless是基于 OpenFaaS 实现的最先进的服务器无感知推理系统，使用CUDA MPS对GPU进行函数分区。

- Astraea：Astraea是一个面向微服务的QoS感知GPU管理系统。提出了一种基于CUDA IPC的自动缩放GPU通信框架。

- Stream-only：Stream only是在stream上运行函数而不进行任何优化的原生版本。

#### Metrics

- total GPU memory footprint
- throughput
- end-to-end latency

#### 实验一：整体性能

- 内存占用: StreamBox 减少GPU 内存使用至多 82%，因为**减少了重复的GPU runtime**
  <img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001165445594.png"  />
- 高吞吐量：StreamBox将系统吞吐量提高了5.3X-6.7X，受冷启动影响小
  <img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001165657368.png"  />
- 低延迟：StreamBox可以保证SLO并减少端到端延迟，更有效的I/O，内存分配、通信
  <img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001165858409.png"  />

#### StreamBox的优化

- 自动伸缩内存池：延迟分配隐藏了内存分配开销，自动内存伸缩使得StreamBox的内存池与实际内存需求非常匹配。
  <img src="https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241001170452756.png"  />

- GPU内通信：降低94%通信开销，StreamBox通过内存压力感知技术最大限度地提高了通信缓冲区容量，并采用了自适应的GPU间数据移动。

  ![image-20241025154304540](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241025154304540.png)

- 细粒度PCIe带宽共享：由于我们使用流水线模型加载方法，模型加载开销与计算重叠，导致延迟只有几毫秒。与Stream相比，StreamBox将延迟降低了91%，这是因为流之间高效的I/O共享。与MPS相比，StreamBox将延迟降低了80%，因为固定内存缓冲区隐藏了分配固定内存的开销。
  ![image-20241025154417549](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241025154417549.png)

#### StreamBox的开销

- API转发：1us-3us，当函数的并发量增加到48（CUDA MPS中单个GPU上的最大并发量）时，它保持在3us-5us的范围内。大多数GPU API都是异步的，这意味着GPU API挂钩和转发的开销可以隐藏在执行中。
  ![image-20241025154433703](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241025154433703.png)
- 离线分析：离线分析只会产生一次性成本，并且可以在以后的请求中重复使用。离线分析和转换的平均时间分别为36秒和4秒。

