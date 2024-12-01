by Yikun Gu
### Pre-Warming is Not Enough: Accelerating Serverless Inference With Opportunistic Pre-Loading

#### 作者

![image-20241108142113938](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108142113938.png)

![image-20241108143105545](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108143105545.png)

[TOC]

#### Background

服务器无感知函数通常经历三个阶段：

​	1）容器预热，

​	2）加载依赖项（如Python库），

​	3）处理查询。

​	对于服务器无感知工作负载，容器预热在冷启动中占主导地位，而加载依赖项的时间成本可以忽略不计。因此，预热方法非常适合这些函数。

​	然而，我们观察到，对于机器学习推理函数，加载依赖项的时间花费——这超出了预热策略的范围——相当显著。

![image-20241108144250148](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108144250148.png)

​	以往研究：无法完全缓解ML组件加载阶段。

​	为了全面加速机器学习推理函数并实现最小化的端到端延迟，我们的目标是进一步超越预热——提前将机器学习模型等工件预加载到容器和GPU实例中。

##### Challenges

- 1） 预加载的内存成本很高。
- 2） 预加载必须避免任何额外的函数启动开销。



​	本文提出了InstaInfer，一种面向服务器无感知推理任务的机会性预加载系统，以应对这些挑战。为了在最小化加载延迟与避免内存浪费之间取得平衡，InstaInfer仅在平台创建的现有预热容器和GPU实例中预加载函数，而不主动预留内存。为了持续提供最佳的函数加速，InstaInfer通过动态加载和卸载函数来高效利用空闲资源。此外，InstaInfer兼容现有的预热和保持活跃（keep-alive）方案，避免干扰容器的创建或移除策略。



#### Motivation

##### 服务器无感知函数生命周期

​	1） 容器预热，

​	2）ML工件（例如库和模型）加载，

​	3）ML推理。

![image-20241108145755208](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108145755208.png)

##### 容器加热与ML工件加载

​	预热已经有很多研究

​	机器学习工件的加载阶段是服务器无感知机器学习工作负载特有的，因为神经网络模型和其依赖库的体积不断增大，导致加载时间较长。

​	加载阶段主导了服务器无感知机器学习推理请求的端到端延迟，但为一般服务器无感知工作负载设计的“冷启动”解决方案忽略了这一点。因此，我们认为预热对于服务器无感知推理函数来说是不够的。

##### 预加载必要性

测试环境：4个A10

workload：我们滑动整个Azure trace，并随机选择八个函数trace来构建工作负载。

baseline：Histogram, FaasCache, Pagurus, inside OpenWhisk



​	现有的预热方法减轻了OpenWhisk上的warming延迟。然而，加载机器学习组件在推理前的总体延迟中占68%以上，而只有25%用于warming，只有6%用于推理。因此，现有的方法严重忽视了服务器无感知推理任务的预加载机会。相比之下，InstaInfer将整个工作负载的时间减少了55%以上，表明预加载显著降低了整体延迟。

![image-20241108150859722](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108150859722.png)





#### An Overview of InstaInfer



##### objectives

- 即时推理：最大限度地减少机器学习推理调用的整体端到端（E2E）延迟
- 零浪费：仅利用现有容器和GPU实例中的空闲容量来预加载函数
- 对提供商透明：预加载应避免与平台固有的预热机制发生冲突。

##### challenges

- 如何在有限的空闲容器和GPU实例的情况下最大限度地提高加速性能？
  由于只有空闲的容器和GPU实例，我们无法同时预加载所有函数。我们必须识别并选择具有高延迟改进潜力的函数，并将其准确分配给每个容器实例。
- 在预加载函数时，如何避免额外的资源开销？
  将库和模型保存在容器中可能会占用大量内存。我们必须在内存浪费和更多预加载之间寻求平衡，以实现最佳加速。
- 如何在不产生额外函数启动开销的情况下启用预加载？
  服务器无感知函数通常具有关键的延迟要求。例如，Azure functions上超过50%的函数在不到一秒钟的时间内执行。我们必须以轻量级和透明的方式设计预加载过程，以避免任何额外的函数启动开销。

##### InstaInfer’s System Architecture

为了在资源约束下实现最佳加速，我们设计了一种安全的实例共享机制，允许将多个函数同时预加载到单个容器中并共享GPU。InstaInfer包括三个主要组件：主动预加载器、预加载调度器和容器内管理器。

- Proactive Pre-Loader利用平台预热机制的预测模型来预测函数调用到达时间。然后，预测结果用于确定何时预加载每个函数。为了实现集群范围内的加速，当收到请求时，它会将请求路由到预加载函数的工作节点。
- Pre-Loading Scheduler在每个工作节点上运行，并将需要预加载的函数分配给适当的容器和GPU。为了随着时间的推移保持最佳的加速，它根据平台预热和保持活动策略触发的工作节点的容器创建和删除事件动态地做出预加载和卸载决策。
- Intra Container Manager独立操作每个函数的加载和卸载执行。我们设计了一个三层安全保护机制，以确保共享同一容器的每个预加载函数的安全性和隐私性。

![image-20241108152432586](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241108152432586.png)



#### Proactive Pre-Loader

​	我们设计InstaInfer是为了在调用到达之前机会主义地预加载一个函数，并在预测错误的情况下卸载该函数以允许其他预加载。

##### Function Pre-Loading and Offloading

​	两个thresholds： $P_{load}(f),P_{offload}(f)$

​	如果概率达到$P_{load}(f)$，则立即预加载该函数。相反，如果函数在很长一段时间内未被调用而保持预加载状态，使得概率超过$P_{offload}(f)$，InstaInfer会识别出预测不正确，并卸载该函数以释放资源用于预加载其他函数。

​	使用过时的数据会严重降低预测的准确性。为了提高预加载的准确性，我们使用滑动窗口来捕捉每个函数的时间变化，并将预测与最新数据对齐。它与各种预测模型兼容，因为我们只调整时间范围而不改变底层模型。

​	使用泊松分布，$\mathcal W$ 为窗口大小，$T_w$ 为窗口中第一个和最后一个激活请求之间的时间请求到达率为$\lambda_f=\cfrac{W}{T_w}$，则下一个请求的到达时间的概率分布为：$F(t;\lambda_f)=1-e^{-\lambda_ft},t>0$。

​	则
$$
T_{load}(f)=-\cfrac{1}{\lambda_f}\ln(1-P_{load}(f))\\
T_{offload}(f)=-\cfrac{1}{\lambda_f}\ln(1-P_{offload}(f))
$$
​	我们将默认的$P_{load}(f)$和$P_{offload}(f)$分别设置为$6%$和$94%$。

![image-20241109122553382](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109122553382.png)

#### Pre-Loading Scheduler

​	我们设计了一个预加载调度器，可以动态选择并将函数分配给适当的实例，以实现最佳加速。为了随着时间的推移优化性能，调度器自适应地调整预加载策略以适应变化。

##### Latency-Aware Function Mapping

​	为了在最大加速和避免额外成本之间取得平衡，我们提出了一种实例共享机制，允许将多个函数同时预加载到单个容器中，直到其空闲内存耗尽，而它们的模型共享同一GPU。

​	多背包问题：该策略通过根据容器容量和节省的延迟确定是否放置函数，$DP[i][j]$ 定义为 $j$个容器中$i$个函数的最大延迟节省（函数到达概率和加载延迟的乘积）。

​	$DP[i][j] = \max(DP[i-1][j], DP[i-1][j-1]+latenct\_savings(i))$ **不合理**

​	类似的方法也用于pre-transfer to GPU。

![image-20241109140039910](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109140039910.png)



#### Intra-Container Manager

​	Intra-Container Manager与调度器交互，以控制函数的进程级执行，包括加载、卸载和模型传输。此外，对于同一容器中的函数，它确保没有资源冲突，并维护安全性。

##### Pre-Loading Management

​	由于每个容器都包含多个函数的预加载进程，设计原则遵循三个步骤：等待未来的调用并将其转发给相应的进程，终止所有与传入调用无关的进程，并保证每个函数的安全性和隐私性。在收到来自调度器的预加载消息后，管理器执行函数代码以加载库和模型。然后，它根据调度器的决定将模型传输到容器的相应GPU。加载后，进程进入阻塞状态，等待未来的调用。

​	预加载函数A和B后，在收到函数A的调用后，管理器将请求转发到函数A进程的输入管道，唤醒进程开始推理并返回结果。为了避免内存抢占并保证函数的安全性，函数a调用的到达会立即终止所有其他预加载进程并清除其内存分配。这种设计确保被调用的函数在干净和隔离的环境中运行。

​	同样，在从调度器接收卸载消息时，管理器终止相应函数的进程并擦除所有相关数据以保护用户隐私。虽然函数被用作黑盒，但用户代码只需要稍作更改即可将模型和库文件暴露给InstaInfer。我们提供两种具有不同目标的修改选项：

- 最大透明度。正如下面的Python代码片段所示，开发人员只需要修改两行代码：首先，用InstaInfer API替换模型加载行（`torch.load`），以暴露模型文件的路径。其次，在加载用于监听调用的模型后添加`sys.stdin.readline()`行。收到请求后，函数流程将恢复。

  ```python
  # Original 
  model.load_state_dict(torch.load(model_path)) 
  inference () ... 
  
  # InstaInfer 
  model.load_state_dict(InstaInfer.load(model_path))
  sys.stdin.readline() 
  # wait for request 
  inference () ...
  ```

  

- 最大限度的隐私。如果首选非侵入式预加载，开发人员可以简单地实现类似于Azure预热触发器的LOAD函数来保存预加载内容。管理器调用LOAD API来执行预加载，而不访问任何函数特定的数据。

![image-20241109141509585](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109141509585.png)

##### Privacy & Security Guarantee

​	由于多个函数的代码和数据存储在同一个容器中，因此有必要保证每个函数的隐私和安全性。InstaInfer提供了一种三层安全保护机制。

- 在用户层，只有属于同一用户的函数才能预加载到一个容器上。
- 在流程层，如图6所示，当函数A的调用到达时，同一容器中的所有其他函数都被卸载。他们的数据和代码会立即被删除。
- 在操作系统层，每个函数的预加载过程和数据都分配了一个由Linux权限域和权限控制管理的唯一非root用户。与此同时，通过chroot监狱等监狱技术加强了隔离。这些设计确保了一个函数的进程被限制访问内存和磁盘上其他进程的数据。操作系统级隔离还避免了函数之间的库版本冲突，因为每个函数的库都是隔离的，并存储在其特定Linux用户的路径下。
- 此外，对于实现最严格的安全保证，即完全禁止容器共享，只允许容器预加载一个函数，InstaInfer仍然明显优于现有的预热方法。

#### Evaluation

##### Implementation

- 基于 Apache Openwhisk
- ML 推理框架：Pytorch
- Proactive Pre-Loader。我们在OpenWhisk的负载均衡器模块中实现了主动预加载器，所有调用都经过该模块。主动预加载器记录调用的时间戳，从而更新每个函数的预测。
- Pre-Loading Scheduler：OpenWhisk在每个节点中运行一个容器池模块来管理每个容器的创建和删除。我们在这个模块中实现了调度器，这样调度器就可以获取预加载所需的所有信息。调度器通过HTTP请求向内部容器管理器发送加载和卸载消息。为了使容器的资源限制与调用函数的配置相匹配，调度器在运行Docker容器时使用–memory、–cpu和–gpu标志指定限制。
- Intra-Container Manager：我们在每个容器的代理中实现了管理器，用于与OpenWhisk通信。我们修改Action Proxy模块以接收来自调度程序的消息。我们修改Executor模块以执行加载和卸载。每个预加载的函数都作为一个独立的进程运行。
- GPU support：由于所有函数都在Docker容器中运行，我们应用了NVIDIA container toolkit，该工具包可以让容器在不进行任何额外配置的情况下使用CUDA设备。为了提高GPU资源利用率，**我们使用NVIDIA MPS将GPU划分为多个函数**，并控制每个函数的GPU限制。



##### Experiment Settings

- Testbed：三个OpenWhisk集群：
  - Single-node CPU cluster：AWS m5.16xlarge EC2实例上的单节点CPU集群，具有64个Intel Xeon Platinum-8175 CPU内核和256 GB内存。我们对该集群进行了E2E延迟评估、与基于快照的解决方案的比较、消融研究、敏感性分析和可扩展性测试。
  - Single-node GPU server：AWS g5.12xlarge EC2实例上的单节点GPU服务器，具有48个CPU内核、196 GB内存和4个NVIDIA A10 GPU。我们对该集群进行了E2E延迟评估和内存成本评估。
  - Multi-node cluster：多节点集群，包括一个控制器节点和四个工作节点，每个节点都是一个AWS m5.8xlarge EC2实例，具有32个CPU内核和128 GB内存。我们对该集群进行E2E延迟评估、1000个函数的大规模评估和预测评估。
- Workloads：为了近似真实世界的调用模式，我们对Azure Function trace中的调用进行采样，这些跟踪是在生产环境中收集的。我们扫描14天的Azure调用trace文件，随机选择8个不同的4小时trace，以满足每个基准函数的变异系数（Coefficient of Variation, CoV）要求。然后，每个trace都被映射到一个推理函数，该函数在评估过程中调用。为了通用起见，我们根据CoV定义了三种模式：可预测（CoV<1）、正常（1<CoV<4）和突发（CoV>4）。
- Models and Libraries：
  - ML framework：PyTorch
  - ML model：CV & NLP, 45MB-549MB
    - AlexNet
    - Inception_V3
    - ResNet18
    - ResNet50
    - Resnet152
    - VGG19
    - GoogleNet
    - Bert-Base
- Baselines:
  - OpenWhisk: OpenWhisk的默认保持活动策略，在调用后使每个容器保持活动状态固定10分钟。
  - Histogram Policy: 一种基于histogram的容器缓存方法，通过预测函数调用的间隔到达时间来动态确定何时预热容器以及容器保持存活多长时间。我们在OpenWhisk内部实施了Histogram Policy。
  - FaaSCache：提出了一种贪婪的双重保活缓存策略来保持函数的存活。我们的评估在OpenWhisk中重用了FaaSCache的开源代码库。
  - Pagurus：Pagurus通过将其他函数的空闲容器“借给”被调用的函数来避免冷启动。
  - REAP：REAP是一种基于快照的冷启动缓解方法，它将函数完成状态作为快照存储在磁盘上。
  - Azure Function with warmup trigger ：带有预热触发器的Azure Function允许在扩展新实例的同时预加载用户定义的内容。
- Evaluation Metrics：
  - End-to-End (E2E) latency
  - Warming+Loading latency
  - Preloading rate：函数已预加载的调用与总调用的比率
  - Speedup
  - Memory cost: 平台运行整个工作负载的CPU和GPU内存消耗。

##### Reducing E2E Latency

​	在single-node cluster上实验。

​	与预热baseline和普通OpenWhisk相比，将InstaInfer与baseline解决方案集成可以减少高达86%的E2E延迟和93%的升温+加载延迟，因为InstaInfer通过库和模型预加载有效地减轻了延迟。

![image-20241109151632000](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109151632000.png)

​	表1显示了每个baseline的平均E2E延迟、升温+加载延迟、加速和预加载率。InstaInfer+*在每个指标上都优于每个相应的baseline。InstaInfer+Pagurus由于有更多空闲的容器用于预装载，因此性能最佳。这是因为Pagurus移除的容器更少，而保持的容器比其他baseline更多。

![image-20241109152235959](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109152235959.png)

​	为了进一步探索E2E延迟的减少，我们展示了InstaInfer和每个baseline运行正常工作负载的E2E延迟累积分布函数（CDF） Fig 8，结果表明，InstaInfer可以有效地加速工作负载，而不会增加尾部延迟。

![image-20241109153533771](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109153533771.png)

​	为了更直观地展示InstaInfer的加速效果，我们在图9中展示了运行“正常”工作负载的Pagurus和InstaInfer+Pagurus的E2E延迟的时间分解。在这种情况下，选择Pagurus是因为它的性能优于Histogram和FaaSCache。图9显示，InstaInfer+Pagurus不仅消除了升温阶段，还消除了大多数调用的库和模型加载阶段。

![image-20241109154022246](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109154022246.png)



##### InstaInfer GPU Evaluation

​	为了展示在CPU内存和GPU内存中机会预加载的好处，我们评估了在具有4个NVIDIA A10 GPU的GPU集群中使用InstaInfer的工作负载的E2E延迟。如图10所示，将InstaInfer与每个baseline集成可以显著降低每个推理函数的平均E2E延迟，最多可达93%。与第7.3节中基于CPU的InstaInfer相比，带有GPU预加载的InstaInfre进一步提高了函数执行时间成本，因为它减轻了CUDA运行时初始化和模型交换延迟。![image-20241109154602899](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109154602899.png)

##### Memory and Monetary Cost

​	我们评估了InstaInfer、baseline预热方法和初始预加载的成本，同时运行相同的Azure trace工作负载。在评估中，InstaInfer与每个baseline相结合。在OpenWhisk预加载baseline中，每个容器只能容纳一个预加载函数。为了实现与InstaInfer相同的加速性能，主动创建了更多的容器进行预加载。如图11所示，InstaInfer+*的容器和GPU内存消耗与单独的相应baseline几乎相同。这是因为InstaInfer只重用baseline方法创建的空闲容器，而不主动创建新容器。因此，InstaInfer不会产生额外的资源成本。相比之下，为了实现类似的加速性能，OpenWhisk预加载创建的容器比InstaInfer多，与InstaInfer相比，最多只能产生2.4倍的内存成本和2倍的GPU成本。

![image-20241109155448343](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109155448343.png)

​	然后，我们使用Azure Function定价模型评估运行上述4小时工作负载的成本。如图12所示，InstaInfer+*的货币成本与单独的相应baseline几乎相同。尽管根据图7，Azure高级计划在几个函数上实现了比InstaInfer更低的E2E延迟，但其费用是其他方法的20倍。

![image-20241109155602843](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109155602843.png)

##### Multi-Node Evaluation

​	我们通过在多节点集群上进行实验来评估InstaInfer的可扩展性。我们使用第7.3节中的相同基准、指标、baseline和工作负载来评估E2E延迟。图13显示，将InstaInfer与baseline集成可以减少高达87%的E2E延迟。在多节点集群上评估的性能与在单节点集群上观察到的结果相似。这种一致性表明，InstaInfer可以有效地为分布式集群中的各种工作负载保持低加载延迟。表2详细列出了每个baseline的平均E2E延迟、升温+加载延迟、加速和预加载率。数据显示，InstaInfer+*在所有指标上始终优于现有baseline。

![image-20241109155736678](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109155736678.png)

![image-20241109155804180](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109155804180.png)



##### Comparisons with Snapshot Methods

​	我们在同一设置中分别在InstaInfer、REAP和vanilla OpenWhisk中使用小（ResNet18）、中（Inception_v3）、大（Bert-Base）模型评估了三个基准ML推理函数的E2E延迟。图14显示，REAP的表现优于OpenWhisk。InstaInfer的性能比REAP进一步提高了1.5到2.5倍。InstaInfer优于REAP的原因是InstaInfer不需要将快照从磁盘加载和还原到内存。由于REAP的快照都存储在磁盘中，当请求到达时，必须将快照读入内存并恢复到进程中，从而引入额外的延迟。根据实验结果，由于模型和库文件的大小较大，推理函数的延迟很高（300-600ms）。相比之下，根据我们在第7.12节中的测量，InstaInfer将函数保存在内存中，实现了可忽略的延迟（5-14ms）。

![image-20241109155942483](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109155942483.png)

##### Prediction Performance Evaluation

​	为了评估InstaInfer的稳健性，我们选择了四种预测模型：泊松分布、基于Histogram policy的预测、随机森林（RF）和自回归综合移动平均（ARIMA）建模。每个模型都用于决定何时加载和卸载函数。我们分别从可预测、正常和突发工作负载中随机选择200个函数trace。如表4所示，泊松在可预测和正常工作负载中表现最佳，而Histogram在突发工作负载中性能最佳。InstaInfer预加载了40%以上的函数，并将工作负载速度提高了1.5倍以上。

![image-20241109160150833](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109160150833.png)

##### Ablation Study

​	我们在单节点集群上进行了消融实验，以评估主动预加载器和预加载调度器的有效性。对InstaInfer的三种变体进行了评估，并与Histogram Policy、Pagurus和FaaSCache进行了比较：

- InstaInfer_NP: InstaInfer without the Proactive PreLoader.
- InstaInfer_NS: InstaInfer without the Scheduler.
- InstaInfer_NPS: InstaInfer without either the Proactive Pre-Loader or Scheduler.

​	图15显示了从Azure随机选择的2小时“正常”trace下E2E推理延迟的CDF。无论使用何种预热方法，InstaInfer始终优于其他变体，因为它充分利用了主动预加载器和调度器。这两个组件之间的协同作用确保了最大限度地减少加载延迟，尽管调用模式和空闲容器的数量发生了动态变化。平均而言，与InstaInfer_NP、InstaInfer_NS和InstaInfer_NPS相比，InstaInfer将工作负载加速了1.16-1.28倍、1.21-1.49倍和1.48-1.73倍。

![image-20241109160902224](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109160902224.png)

##### Scalability and Overhead

​	为了评估InstaInfer的可扩展性，InstaInfer+Pagurus的工作负载越来越重，从每分钟10到180个请求不等。性能如图17所示。InstaInfer在不同尺度上始终优于Pagurus。然后，我们通过改变容器池的大小来评估InstaInfer在资源预算受限的情况下相对于其他baseline的性能。如图18所示，InstaInfer在不同内存预算下始终优于其他baseline，显示出更强的鲁棒性。

![image-20241109160955279](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241109160955279.png)



​	接下来，我们报告每个组件的延迟和资源开销。主动预加载器在最重的工作负载下引入了不到3毫秒的额外延迟。容器内管理器引入了2ms到11ms的延迟开销，这是由于在调用到达时清除其他预加载进程的内存所导致的内存抢占。此延迟因要卸载的函数的内存占用而异。由于调度器的预加载和卸载决策与服务调用是异步的，因此不会导致延迟开销。与节省的延迟（1500-5000ms）相比，额外的延迟（5ms-14ms）可以忽略不计。处理更少的调用时，开销会更低。

​	在最重的工作负载下，主动预加载器消耗不到0.3个CPU核心和72MB内存；调度器消耗0.3个CPU核和135MB内存；容器内管理器消耗0.1个CPU核心和9MB内存。与工作负载相比，InstaInfer所有组件的总体资源开销可以忽略不计。
