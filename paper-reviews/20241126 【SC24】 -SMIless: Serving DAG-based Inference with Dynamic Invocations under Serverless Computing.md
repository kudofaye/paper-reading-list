by Yikun Gu

### SMIless: Serving DAG-based Inference with Dynamic Invocations under Serverless Computing

![image-20241119182702577](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241119182702577.png)

![image-20241119182851089](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241119182851089.png)

#### Background

ML serving中，一个应用（intelligent personal assistant (IPA)）包含多个推理模型，各种模型通常以并行、顺序或两者的组合进行组织，可以表示为**有向无环图（DAG）**。

![image-20241119185446178](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241119185446178.png)



机器学习推理应用程序经常遇到各种各样的服务请求到达模式，利用服务器无感知计算，应用程序可以通过动态启动不同的函数来精确地调整其资源利用率，这些函数通常在收到调用请求时或接近收到调用请求时触发，具体取决于为缓解冷启动而实施的预热和保活机制。

随着向服务器无感知计算的转变，在生产集群中为机器学习推理提供服务的底层硬件资源正在经历越来越大的**异构性**，这包括各种各样的加速器，如GPU、TPU和FPGA，所有这些都是为了提高机器学习应用程序的性能而量身定制的。虽然高端加速器可以有效地减轻推理开销，但它们也带来了巨大的成本负担。因此，这种异构性提出了一种设计上的**trade-off**，以平衡性能和货币成本。

![image-20241125192622303](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241125192622303.png)

为在利用异构硬件的服务器无感知平台上部署具有DAG拓扑的ML应用程序制定高效的资源配置策略带来了重大挑战。一方面，一个函数的资源配置会影响DAG应用程序中所有后续函数的冷启动管理，从而产生**级联效应**。因此，设计所有函数的资源配置和冷启动管理策略以实现最佳性能变得至关重要。另一方面，调用模式的高度动态性意味着适用于一个调用的策略可能不适用于另一个调用，因此需要考虑未来调用的全局优化，从而增加了任务的复杂性。

关于异构环境中基于DAG的应用程序的资源分配的传统文献忽略了解决冷启动问题，导致资源浪费和无法满足端到端（E2E）延迟的SLA要求。现在的研究主要强调单独指标的优化或者完全无视DAG结构。

#### Motivation

在异构服务器无感知计算下，平衡E2E延迟和使用DAG为ML应用程序提供服务的成本之间的最佳权衡带来了重大挑战，需要在所有函数之间进行全局协同优化。

##### Challenges

主要挑战源于一个函数的资源配置对后续函数的冷启动管理引发的级联效应。

- 当某个函数的推理延迟较大时，为后续函数提供了更长的重叠窗口。这种重叠窗口可以在前一个函数运行期间初始化后续函数，从而减少了不必要的“保持活跃（keep-alive）”时间。保持活跃时间越短，运行成本也就越低。
- 然而，重叠窗口的大小不仅取决于前一个函数的推理延迟，还受到后续函数的初始化时间限制。后续函数的初始化时间本身又与其资源配置紧密相关。资源配置会影响初始化速度，从而进一步影响后续函数的运行时表现。

第二个挑战来自推理应用程序中调用模式的动态特性。

##### Limitation of Existing Works

Orion(OSDI'22)：假设函数资源配置时采用“正确预热（right pre-warming）”策略，即每个函数的执行时间能够与其后续函数的初始化时间完全重叠。这一假设能显著简化跨函数优化的复杂性，但仅在两个函数调用之间的间隔时间较长时成立。如果调用间隔较短（如高频请求场景），则这一策略可能导致次优结果，因为资源分配未能有效适应快速到达的多次调用。

IceBreaker(ASPLOS'22)：对每个函数的资源配置和冷启动策略进行独立管理，而未能充分利用 DAG（有向无环图）拓扑结构的整体特性来实现协同优化。这种独立管理的方法忽略了函数之间的依赖关系，可能导致全局性能和成本优化的不足。

![image-20241120163119483](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241120163119483.png)

#### SMILESS ARCHITECTURE

##### Key Design Ideas

- 利用**自适应预热**来管理动态调用。SMIless为DAG应用程序中的函数引入了一种创新的自适应冷启动策略。此策略有助于根据给定的资源配置和调用到达率计算所有服务器无感知函数的预热大小。这种能力使SMIless能够有效地形式化复杂DAG应用程序的E2E延迟和总成本。
- 通过路径搜索形式化解决级联效应。SMIless通过将问题重新定义为多叉树中的路径搜索来启动其优化方法。DAG内所有函数的自适应冷启动策略和资源分配策略的每个不同组合都被视为一个唯一的节点，这些组合中的变化表示为边。SMIless系统地遍历这棵树，优先考虑成本最小化的组合。这种战略探索使SMIless能够有效地解决级联效应，无需进行详尽的探索即可接近最佳解决方案。

##### System Overview

- Offline Profiler：在后台运行，系统地从部署函数的不同配置中收集执行和初始化时间。利用这些trace，Offline Profiler构建了一个定制的模型，用于分析推理时间和初始化时间。
- Online Predictor：由两个部分组成：the Inter-arrival Time Predictor and the Invocation Predictor。在线预测器收集有关应用程序调用模式的信息，然后使用到达间隔时间预测器构建特定于每个应用程序的模型来预测到达间隔时间，这有助于管理冷启动场景和硬件配置。调用预测器预测每个时间窗口内潜在调用次数的上限，使Auto-scaler能够有效地处理突发工作负载。
- Optimizer Engine：一旦收到开发人员的ML请求，SMIless就会使用优化器引擎通过三个关键模块生成最佳的初始化和执行策略：the Workflow Manager, Strategy Optimizer, and Autoscaler。工作流管理器将DAG分解为多个简化的子图。从策略优化器中获取每个子图的优化策略后，工作流管理器智能地组合这些策略以最小化总体成本。随后，自动缩放器根据预测的调用次数和函数的硬件配置动态操作，以控制所有推理函数的缩放。

![image-20241120165915591](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241120165915591.png)

#### OFFLINE PROFILING AND ONLINE PREDICTION

##### Function Profiling

工具：Prometheus

1. Profiling initialization time：
   - 初始化：下载镜像、初始化容器，但**初始化时间由于资源竞争会出现波动**，Offline Profiler计算均值$\mu$和标准差$\sigma$，使用$\mu+n\times\sigma$作为初始化时间的稳健的估计。（**n？**）
   - 由于初始化GPU，初始化时间会比预估的更长，Offline Profiler对CPU函数和GPU函数进行估计。
2. Profiling inference time：
   - 配置内存略超过knee point
   - 使用MPS切分GPU，最小GPU unit是10%
   - 考虑batch size B，# of CPU cores，% of GPU
   - CPU函数：
     Inference time=$\lambda_c\times B\times(\cfrac{\alpha_c}{\#\ of\ CPU\ cores}+\beta_c)+\gamma_c$
     其中$\alpha_c$是所需计算量，$\beta_c$是CPU执行时的额外时间，如上下文切换、IO、cache miss，$\lambda_c$是多核CPU的性能下降系数，$\gamma_c$是网络传输时间。
   - GPU函数：
     Inference time=$\lambda_g\times B\times(\cfrac{\alpha_g}{\%\ of\ CPU\ cores}+\beta_g)+\gamma_g$
   - 通过对每个函数进行曲线拟合，得到参数$\{\lambda_i，\alpha_i，\beta_i，\gamma_i\}_{i=c，g}$。

##### Online Prediction

在收到用户的调用请求后，SMIless中的网关将此信息转发给在线预测器，以便在指定的时间窗口内计算每个应用程序收到的调用次数，在SMIless中，该时间窗口设置为1秒。

1. Predicting the invocation number
   调用预测器预测下一个时间窗口内每个应用程序的预期调用次数。为了防止低估和避免SLA违规，调用预测器采用了一种分类方法，而不是回归方法，利用LSTM模型。具体来说，预测器将预测空间划分为多个**桶**，并以桶的上限作为预测。桶大小被设定为等于应用程序函数的最小批大小。
2. Predicting the inter-arrival time
   指两次连续的非零调用之间的时间间隔。



#### SMILESS RESOURCE OPTIMIZATION

##### Co-optimization Framework

保证SLA同时减少cost

$C_{k}\left(\star_{k}, \triangle_{k}\right)=E_{k}\left(\star_{k}, \triangle_{k}\right) \cdot U\left(\star_{k}\right) .$

$\star_k$是函数资源配置，$\triangle_k$是函数冷启动策略，$U(\cdot)$是单元执行cost，$E_k(\cdot)$是执行时间，$C_k(\cdot)$是执行cost。

优化问题：

$\min\limits_{\{\vec{\mathcal{X}}, \vec{\mathcal{\varphi}}\}}\sum\limits_{k=1}^NC_k(\star_k,\triangle_k),\ \ subject\ to,\mathcal{L}(\vec{\mathcal{X}}, \vec{\mathcal{\varphi}})\leq SLA.$

$\vec{\mathcal{X}}=\{\star_1,\star_2,...,\star_N\}$

$\vec{\mathcal{\varphi}}=\{\triangle_1,...,\triangle_N\}$

$\mathcal{L}(\vec{\mathcal{X}}, \vec{\mathcal{\varphi}})$获取$\vec{\mathcal{X}}\times \vec{\mathcal{\varphi}}$下的E2E latency，该优化问题为NP-hard

##### Adaptive Cold-Start Management

该设计旨在根据DAG拓扑和调用模式动态更新每个单独函数的预热大小。基于这一策略，SMIless能够计算延迟函数$\mathcal{L}(\vec{\mathcal{X}}, \vec{\mathcal{\varphi}})$。

###### Adaptive pre-warming

![image-20241122095837557](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122095837557.png)
$T_k$：初始化时间，$I_k$：推理时间

- case 1：调用到达率低
  $F_2$推理完成后，卸载$IT-T_2-I_2$时间后再初始化

  若$I_2+I_1\lt SLA$且$T_2+I_2\lt IT$时，$\mathcal L(\vec{\mathcal{X}},\vec{\mathcal{\varphi}})=I_2+I_1,C_2=(T_2+I_2)\cdot U(\star_2)$

- case 2：调用到达率高
  当$T_2+I_2\ge IT$，两种策略处理接下来的调用
  在上次调用后终止实例，并在下一次调用到达之前创建一个新的实例，执行cost为$(T_2+I_2)\cdot U(\star_2)$
  或在上一次调用后保活函数，cost为$IT\cdot U(\star_2)$。

当面临严格的SLA要求时，SMIless应该为F2优先考虑更高级的硬件配置。该选择将导致较小的推理时间I2，随后导致较大的预热窗口大小。
上述比较还意味着，当SLA要求宽松时，为每个函数选择低端硬件并使相应实例长时间保持活动状态的可能性更高，从而最大限度地减少初始化的发生。

###### Adaptive batching

这是指在短时间内有多个调用到达的情况，在每个函数实例内顺序处理调用很容易导致SLA违规。
Auto-scaler采用adaptive batching，此时推理时间上升，SMIless动态采用更好的资源配置，如果还是不能满足SLA，启用多实例来处理。

##### Co-optimization Algorithm Design

the Strategy Optimizer通过TopK路径搜索结合BFS和DFS来优化所有函数的硬件配置和相关冷启动管理策略。

多叉树为搜索空间，每个节点代表应用程序中所有函数的硬件配置和相应的冷启动管理策略，从而有效地解决了级联效应。

通过遵循从根节点到叶节点的路径，可以遍历应用程序中的所有函数，从而得到原始优化问题的唯一解决方案。此外，路径上的每个节点都包含一个成本值，该值表示基于当前策略与函数相关的费用。

###### Top-K path search process

![image-20241122104308155](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122104308155.png)

###### Handling complex applications

DAG很复杂时，the Workflow Manager将DAG分解为几个小的只有顺序依赖的子图。每个子图得到一个执行方案，最后组合。

组合过程从遍历DAG图开始，以识别包含并行分支的最小子结构。假设并行分支结构的起始和结束函数分别为$F_s$和$F_e$，工作流管理器遍历包含函数$F_s$和$F_e$的每条路径，以发现推理时间最短的配置，随后将其分配为$F_s$和$F_e$的最终配置。在此之后，工作流管理器会更新这些并行分支上其他函数的配置，以确保每条路径的整体E2E延迟在组合子图之前和之后保持不变。工作流管理器对下一个最小并行子结构重复此过程，直到所有并行子结构都已处理完毕。

###### Complexity Analysis

通过采用DAG分解，策略优化器可以同时处理每个分解的简单图，从而使算法的复杂性取决于最长路径的长度。假设沿着这条最长路径有N个函数，每个函数有M个硬件配置候选，该算法首先识别所有函数的成本最小化配置组合。获得初始配置的时间复杂度为$O(N·M·\log M)$。（**排序？**）

如果应用程序的端到端延迟不符合SLA要求，则算法将继续替换函数的硬件配置，以减少应用程序的延迟。在最坏的情况下，该算法可能需要遍历N个函数中的每一个的所有M个配置，以验证当前配置的端到端延迟是否满足SLA要求。

##### Container Autoscaling

对于预测的下一个interval的请求数量$G$，一个实例中batch $B$个请求，则有$\cfrac{G}{B}$个实例，优化问题：

CPU函数：

$\min\limits_{\{\star_k,B\}}\cfrac{G}{B}\cdot IT\cdot U(\star_k),$

$subject\ to,\lambda_c\times B\times(\cfrac{\alpha_c}{\#\ of\ CPU\ cores}+\beta_c)+\gamma_c\leq I_s$

GPU函数中限制变为：

$\lambda_g\times B\times(\cfrac{\alpha_g}{\%\ of\ CPU\ cores}+\beta_g)+\gamma_g\leq I_s$

自动缩放器采用多线程并行解决所有函数的优化问题。在处理每个优化问题时，它采用双分割法来确定$B$的最优解和相应的配置$\star_k$。

#### SYSTEM IMPLEMENTATION

- 基于：OpenFaaS



#### EVALUATION

##### Experimental Setup

###### Applications

![image-20241122123725424](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122123725424.png)

![image-20241122123804134](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122123804134.png)

- AMBER Alert
- Image Query
- Voice Assistant

###### Load generator

Azure Function Dataset

###### System Settings

8机集群，每台机器都配备了两个52核Intel x86 Xeon Gold 5320 CPU、128GB内存和一个Nvidia RTX 3090 GPU。这些机器通过10GbE NIC连接，并安装了CUDA 11.5和cuDNN 8。

CPU容器有五种不同的规格，配备1、2、4、8或16个内核，对应于AWS c6g系列实例。

使用MPS以10%为单位分配容器的GPU资源。

###### Baselines

- GrandSLAm：GrandSLAm是一个为多阶段机器学习应用程序设计的整体运行时框架，其目标是在保持SLA要求的同时最大限度地提高吞吐量。
- IceBreaker
- Orion
- Aquatope：Aquatope是一个用于服务器无感知工作流的不确定性感知QoS调度器，利用贝叶斯优化来确定最佳资源配置。

##### E2E Performance

![image-20241122124807008](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122124807008.png)

SMIless非常接近最优解（通过穷举搜索确定），重要的是，随着DAG复杂性的增加，SMIless和最优解决方案之间的成本差异会减小。

与IceBreaker相比，SMIless的成本降低了5.73倍，同时保持了SLA合规性，没有任何违规行为。这是因为IceBreaker只考虑不同硬件的调用次数和函数速度，而忽略了DAG结构。

![image-20241122125406736](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122125406736.png)

因此，如图9（a）所示，它导致GPU资源中长期保持活动的实例比例很高，导致GPU上的大多数函数保持活动状态。

由于缺乏有效的协同优化硬件选择和冷启动缓解，Orion的违规率相对较高，约为40%。因此，Orion无法完全利用调用模式来优化不同函数的配置，导致成本是SMIless的两倍。

相反，Aquatope实现了低成本，但代价是SLA违规率高达40%。尽管Aquatope考虑了调用模式并基于DAG结构设计了资源配置，但频繁初始化容器会导致许多冷启动。为了验证这一论点，我们评估了整个实验过程中函数重新初始化的比例，如图9（b）所示，其中Aquatope在所有解决方案中表现出最频繁的初始化。虽然GrandSLAm由于初始化时间较少而表现出较低的E2E延迟，但缺乏冷启动管理导致成本高达SMIless的2.46倍。

![image-20241122125720493](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122125720493.png)

此外，我们还研究了SLA设置对整体执行成本的影响。如图10（a）所示，Orion从严格的SLA设置中获得了最大的好处。具体来说，当SLA超过5秒时，Orion的成本仅为SMIless的两倍。图10（b）提供了相应的SLA违规率。重要的是，应该指出的是，无论SLA设置如何，SMIless都能始终如一地实现最低成本，而不会违反SLA。此外，SMIless在总成本方面表现出稳定的性能，这得益于其路径搜索策略，当SLA要求发生变化时，该策略仅更新少数函数的硬件配置和冷启动策略。



##### Source of Improvement

###### Offline profiling

![image-20241122125923526](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122125923526.png)

当使用平均初始化时间作为测量值时，SLA违规率可达34%。然而，使用SMIless，通过引入3倍的不确定性，可以完全避免SLA违规。

此外，函数推理时间的分析在准确指导协同优化过程中起着至关重要的作用。如图11（b）所示，分析方法具有很高的精度。所有函数的MAPE（平均绝对百分比误差）均小于20%，总体平均值低于8%。值得注意的是，GPU推理时间的预测比CPU推理时间更精确。这种差异归因于CPU上的执行更容易受到干扰。推理时间分析的卓越准确性是通过CPU后端上只有5×5=25个样本（包括21∼25和20∼24个CPU内核的5种批大小）和GPU后端上的50个样本（涵盖10个不同百分比的GPU分配）来实现的。



###### Online prediction

![image-20241122131650236](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122131650236.png)

其次，准确的预测使SMIless能够在高度动态的调用到达的情况下执行可靠的主动资源配置。我们比较了在1小时跟踪上训练并在21小时数据集上测试的不同预测器的结果，该数据集显示调用次数的方差与均值之比大于2。这些预测因素是：

- 1）XGBoost；
- 2） ARIMA，一种被广泛采用的时间序列模型；
- 3） FIP，IceBreaker使用的基于傅里叶变换的方法；
- 4） SMIless，采用一个（或两个）LSTM模块，具有30（128）个隐藏状态用于调用号（到达间隔时间）。

如图12（a）所示，调用次数的预测器实现了3%的低估误差，优于基线，主要归因于其优越的分类方法。为了弥补这一误差，SMIless在原始数量的基础上增加了3%。如果没有这种补偿，SLA违规率将上升到5%。关于高估，SMIless估计，与基于真实调用次数的资源使用情况相比，CPU后端的容器数量和GPU后端的容器数分别略微增加了4.32%和1.85%。

到达时间间隔的预测器表现出令人印象深刻的低sMAPE（对称MAPE），仅为2.45%。此外，由于其具有两个LSTM模块的创新设计，它的高估概率低于0.64%。与采用具有到达间隔时间作为唯一输入的单个LSTM模块（称为SMIless-S）相比，这种配置将高估误差显著降低了10倍。这种高精度使SMIless能够在同一应用程序中实现函数初始化和推理之间的完美重叠。值得注意的是，用ARIMA代替LSTM预测器将导致总体成本增加18%。

###### Co-optimization

![image-20241122131710851](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122131710851.png)

SMIless对改进的关键贡献可归因于资源配置和冷启动管理的有效协同优化。我们评估了SMIless的两种变体：SMIlessNo DAG和SMIless Homo，其中前者忽略DAG结构，并根据到达时间间隔同时预热所有函数实例，而后者仅利用CPU后端进行资源配置。如图13（a）所示，SMIlessNo DAG下的总成本比SMIless高39%，强调了全局冷启动管理在所有函数中的重要性。此外，如图13（b）所示，当SMIless Homo仅考虑同构资源时，应用程序的SLA违规率高达22%。



##### Adaptation to Bursty Arrivals

![image-20241122133024957](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122133024957.png)

![image-20241122133041899](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122133041899.png)

我们对SMIless在处理突发工作负载方面进行了评估。具体来说，我们对一个60秒的时间窗口进行采样，在此期间，工作负载表现出较大的波动。如图14（a）所示，SMIless显示出对工作负载变化的快速响应，Pod的数量根据调用的数量而显著变化。关于具有不同后端类型的容器的比例，图14（b）显示，当调用次数增加时，CPU和GPU的比值也急剧上升。这可以归因于GPU在并行处理批处理调用请求方面更高效，因此在突发工作负载设置中只需要扩展少量GPU实例。

如图15所示，SMIless展示了在做出在线扩展决策时，在降低总体成本和满足SLA要求之间的最佳权衡。具体来说，在这段突发工作负载期间，Auqatope、Orion和Icebreaker的成本要高出1.41倍以上。尽管GrandSLAm由于其有限的资源扩展能力而实现了最低的成本，但它因此导致了SLA违规，违规率高达20%。

##### System Overhead

![image-20241122133054105](https://raw.githubusercontent.com/wty403/pic/main/picture/image-20241122133054105.png)

如图16（a）所示，随着DAG内最长路径的长度增加，使用策略优化器查找策略的开销几乎呈线性增加。具体来说，SMIless可以在20ms内为具有12个函数的DAG的最长路径找到接近最优的策略，与其他路径搜索方法相比，减少了10×∼100×。此外，自动缩放器的优化过程需要不到0.1毫秒的时间来计算最佳缩放结果，如图16（b）所示。这种开销相对较小，突显了SMIless在协调优化资源配置和冷启动管理方面的效率。
