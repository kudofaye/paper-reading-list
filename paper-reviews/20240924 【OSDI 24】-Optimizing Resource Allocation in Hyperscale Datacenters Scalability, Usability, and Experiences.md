# Review：Optimizing Resource Allocation in Hyperscale Datacenters: Scalability, Usability, and Experiences

## 论文作者：

- Neeraj Kumar：Software Engineer, Dropbox

- Pol Mauri Ruiz：

- Vijay Menon

- Igor Kabiljo

- Mayank Pundir

- Andrew Newell

- Daniel Lee

- Liyuan Wang

- Chunqiang Tang

  

## 解释

- Scalability：可扩展性。要实现可扩展性，要解决许多复杂的优化问题，这些都是NP-hard的。Datacenter越扩展，越难高效求解。
- Usability：实用性。大部分资源分配工作的痛点，使用者不能将实际问题转化为形式优化方法所需的精确数学公式。
- Experiences：一些经验。作者分享了一些资源分配工作上的经验。

## 背景

MIP尽管在这类问题中被广泛使用，但是在大规模生产系统中是受限的，由于Scalability和Usability的限制。

- Usability：虽然MIP问题的模型及概念很容易理解，但大多数系统从业者缺乏将实际系统的复杂策略转化为MIP精确数学公式的训练。尽管DCM以及Wrasse针对问题的描述语言做了一些工作，来表达问题和约束，但这些方法的行业采用情况尚未见报道，因此其一般性仍未得到验证，没有充分解决下面描述的可伸缩性挑战。
- Scalability：**赋值变量vi j表示对象i是否被赋值到bin j**。因此，一个指派问题有| O | × | B |个变量。在大规模的云中，问题非常庞大，且大部分分配问题都是NP-hard的。

## 贡献

- 提出了第一个解决广泛的分配问题的框架Rebalancer，针对一系列把objects分配到bins的抽象，给出了解决方案，在生产使用中得到了验证。

- 提出了一套高级规范API，开发人员可以轻松地构建分配问题和约束。

- 使用了DAG图表达问题的目标和约束，问题规模大小为O(| O | + | B |)

- 提出了一种局部搜索算法，利用表达式图G，解决大问题的分配问题。对于小问题，在G上递归调用mipTranslate翻译为MIP模型，使用商业求解器求解。

  

## 模型规范

### 概念

- Dimensions（D）。与每个object和bin相关的数。如：对于server（bin）的D，代表内存容量;对于task（object）的D，代表运行所需内存。
- Scope。bin的层级。如对于数据中心的server，有datacenter或rack的scope。一个scope中的不同bin集合称为scope items。
- Partition。object的层级。partition下有多个object集合，称为group。（如jobs和job）
- 利用率：![image-20240925134547656](https://i.postimg.cc/cLpJt1bV/image-20240925134547656.png)一些变体：如util(rackr, jobj, ObjectCount)，util(bj, D, TIME)。

其中TIME表示一个**时间状态**，AFTER表示当前被分配到bin上的objects，INITIAL表示初始被分配到bin上的objects，STAYED表示bin上存留的初始objects。

还有NEW表示新移动到bin上的objects, OLD表示bin上移出的初始的objects, ANY表示bin上留存过的所有objects。

### 高级Specs API

这些高级spec API优先考虑易用性，并且它里面的低级API表达式（Max，Sum，Ceil，Log，Power...）为新spec开发提供了可能，具备可扩展性。

表达式API的存在方便了specs的实现，只需要几十到几百行代码。

![image-20240925140214699](https://i.postimg.cc/8czkg77r/image-20240925140214699.png)

使用CapacitySpec、GroupCountSpec作为约束，BalanceSpec作为目标。

## 模型表示

将高级Specs API翻译为表达式的递归组合。

例如，scope items上使用约束MinimizeMovementSpec（最小化进出scope items的objects），维度为count。这将产生一个约束。而MinimizeMovementSpec被翻译为util(Sout, count, NEW) + ∑j util(Sj, count, NEW)。



表达式图中：

- 节点，代表表达式
- 各个子图，表示问题中的目标或者约束
- children(v)，表示v的子节点w，有v->w的出边
- type(v)，表示节点上的数学运算符，**children(v)是它的输入**。



一个例子：

![image-20240925140425590](https://i.postimg.cc/G2rhz7rK/image-20240925140425590.png)

lookup表达式：

由于object的D是不受所分配的bin影响的，所以不同的bin和scope item可以重用D，输入规模缩小了Θ(|B|)。

利用VD，一个objects到D的映射，大小为Θ(|O|)，可以把util(Si, D)转为Lookup(Si,VD)，Si是一个scope items，大小为Θ(|B|)。问题规模变为**Θ(|O| + |B|)**。



Tb是目标子图，使用了BalanceSpec，作用到LOOKUP1和LOOKUP2上，即要求它们的利用率平衡。

Tc是约束子图，使用了CapacitySpec，作用到每个server上，要求cpu低于某个限制。



Rebalancer中的**约束都是f ( · )≤0形式的不等式，固可以和Max表达式结合：Max（...）≤0**。



## 模型求解

### 小问题求解

转化为MIP问题，利用MIP求解器求解。

每个表达式执行一个递归的mipTranslate操作。该操作根据type(v)，转化为对应的二进制赋值变量vi j的线性组合。

### 大问题求解

利用局部搜索算法，但是**不保证最优解**。

步骤：

- 生成候选moves，每个move为（oi，bs，bd）。即oi从bs移到bd这个动作。

- 评估moves。（由于移动评估不影响当前状态，因此在选择最佳候选应用之前，可以使用**多个线程**并行地评估候选移动）

- 应用最佳的moves。

  

#### 关于候选moves的生成

共|O| × (|B| − 1)个候选moves，要全部都评估一遍，开销太大。Rebalancer把每次的搜索限制在一个bin，并执行以上步骤，找到最佳moves。

#### 如何确定bin的选择顺序？

定义了hottest bin，即在当前目标和分配下，objects进出对目标贡献最大的bin，探索hottest bin能够保证时间花费在那些改进较大的动作上。

#### 如何寻找hottest bin？

假设叶子节点Sv, Sw, . . . , Sz是它们对目标贡献的贪心顺序（Sv贡献最大）。

若每个Si中只有一个bin，那么hottest bin就是Sv中的bin。

若有多个，例如{1 : {bp, bq}, 2 : {bp, bq, br}, 3 : {bp, br}，4 : {bq, br}}，那肯定在bp, bq中选，而bp, bq同时在2中出现，则看集合3，发现只有bp，那么bp即为hottest bin，并停止。

#### move的移动策略

- SINGLE：对于一个源bin bs，会产生到所有其他bin bd的moves

- SINGLE_GREDDY: 不会像SINGLE那样枚举全部，遇到第一个更优的move时就停止。

- SINGLE_RANDOM: 类似SINGLE,但是bd来自一个随机的小范围bin。

  

还支持更多复杂的策略（如SWAP，KL_SEARCH），一般当改进空间很大时，使用**SINGLE_RANDOM**策略，而改进空间不大时，使用**SINGLE_GREEDY**策略。

也支持自定义的移动策略，如一些独特的领域知识的支撑的策略。

#### 如何评估moves？

只有部分节点可能会受到应用moves移动的影响，我们可以通过只重新计算这些受影响节点的值来显著地加速计算。

提出了一种Bottom-up的改变传播方法，**从这些受影响的叶子遍历到根**，沿途到达的节点是需要重新计算值的节点集合。

#### 更新节点值时如何减小计算量？

可以**根据type(v)，存储一些额外信息**，加速这个重新计算的过程。

例如，对于**Max**操作，分别计算其变化子节点的最大值(记为z1 )和未变化子节点的最大值(记为z2 )。那么，节点的新值就是max( z1 , z2)。

为了有效地计算z2，我们维护了一个**按节点值递减排序**的子节点列表，在这个列表上进行迭代，并在**第一个未改变**的子节点处停止，这个子节点的值即为z2。

计算时间与改变的子节点数c成正比，产生O( c log l)的开销，l为子节点总数。

#### 关于等价类的寻找

我们可以从建模的角度识别等价的对象，从而有效地减少对象的数量。我们只需要探索等价类中最多一个等价对象的移动，这也减少了搜索空间。

**什么是等价类？**

例如：在将任务放置在服务器上的问题中，所有属于同一作业的任务都是等价的，因为它们都以相同的方式影响约束和目标。



**如何判断等价类？**

设A是给定问题的任意可行解，A‘是交换oi和oj所在bin的位置后的解，若所有约束和目标表达式对A和A′的评价都是相同的，那么oi和oj被认为是等价的。

#### 终止条件

一般的，当不能做出任何改进时，终止。但某些情况下，根据目标值，可能可以确定未来是否存在改进，若不能，则提前终止。

- 全局优化：若目标子图的根节点值达到了预估的下界，则认为当前的分配全局最优并终止。

- 局部优化：在局部搜索时，会遇到一些bin，基于它们的任何moves都会违反约束或无法优化目标，论文中称为explored bins，它们的约束下界就是它们的当前值。所有叶节点的约束下界计算完后，我们可以**递归的计算所有表达式的约束下界**，从而得到目标根节点处的约束下界。若**目标根节点的值等于它的约束下界，那么当前分配局部最优并终止**。

  

### 经验与方法

解决优化问题的方法：

1、ad hoc启发式方法。

2、形式化问题规范+形式化求解器。（Rebalancer对于小问题的求解）

3、形式化问题规范+系统启发式求解器。（Rebalancer对于大问题的求解）



经验和建议：

1、需要短暂且可预测的延迟的系统（如高通量、短生命周期的批量作业），建议方法1。

2、需要针对中小型问题提供高质量的解决方案，建议方法2。

3、不需要实时决策的超大规模系统，建议方法3。（更容易添加和改进策略）

## 实验

### 与DCM的对比

DCM: 集群管理中，DCM使用SQL来表示节点放置pods的策略，然后转化为约束满足问题，然后使用Google **OR-tools CP-SAT求解器**进行求解（和MIP类似）。



在Rebalancer中复制了DCM对Kubernetes调度器的实现，并施加了pod间的反亲和，增加调度难度。



使用了**Azure数据集**。所有的实验都是在40个CPU核心和256GB内存的机器上进行的。



该**实验由于一些限制，无法直接比较Rebalancer和DCM在同一台机器上的绝对性能**，但从使用相同数据集和执行相同Kubernetes调度算法，以此来评估可扩展性趋势。



#### Scalability

在两种规模下对Rebalancer进行评估：

1、DCM规模，在1k、5k和10k个节点(类似于DCM论文中的结果)的集群上，每批调度50个pods

2、超大规模，在10k、50k、100k个节点的集群上，每批调度500 ~ 5k个pods

![image-20240925145654825](https://i.postimg.cc/vBz8KrZs/image-20240925145654825.png)

- 对于多达10k个节点的实例，SINGLE _ GREEDY的per-pod调度延迟小于35ms。然而，超过10k个节点后，SINGLE _ GREEDY逐渐变慢，而SINGLE_RANDOM继续保持良好的扩展性。
- DCM论文中解决的最大问题涉及在10k节点集群中调度50个pod，每个pod的调度延迟接近30ms。相比之下，即使在集群规模为100k个节点(是DCM实验的10倍)，批处理规模为5k个pods(是DCM实验的100倍)的情况下，SINGLE _ RANDOM实现了14ms的per-pod延迟。

#### Solution quality

在500个节点上放置10k个来自Azure数据集的pods，将Rebalancer的局部搜索与一个最优的MIP求解器进行比较，目标是最大化放置pods的数量。

MIP和局部搜索都不能放置所有的pods，因为它们的总资源需求超过500个节点的能力。

![image-20240925150745595](https://i.postimg.cc/yN08m4qW/image-20240925150745595.png)

- Azure dataset下，当节点满时，SINGLE _ RANDOM比MIP减少1.4 %的pod，但运行时间是MIP的5.2倍。

  

为了评估在不太可能表现良好的场景中的局部搜索，我们设计了**两个以N为参数的病理案例**，有31N个pod和50N个节点，每个节点的内存为32G。

Pod内存需求在5组数据中服从指数分布：前16N个Pod内存为2GB，后8N个Pod内存为4GB，后4N个Pod内存为8GB，后2N个Pod内存为16GB，最后N个Pod内存为32GB。

唯一的调度所有pods的方式是每个节点上使用的内存总量恰好为32GB，但**由于局部搜索性质，几乎不可能找到**。

- 局部搜索可以放置97 %以上的pods，同时比MIP快200倍以上

### 与POP对比

pop将一个大的MIP问题分解为多个小的子问题，并独立求解，根据需要协调解决方案。

作者实现了Rebalancer中的POP-like的分区MIP求解器。

使用我们的生产数据对服务放置问题的局部搜索和分区MIP进行了比较。

![image-20240925151741150](https://i.postimg.cc/5yK44fzK/image-20240925151741150.png)

局部搜索使用移动类型的组合，包括SINGLE _ RANDOM，SINGLE _ GREEDY和SWAP。局部搜索的速度提高了4倍，满足了覆盖范围和容量充裕性等所有服务级别的要求，同时其解的质量比分区MIP差不到0.6 %。

### 证明hot bin的处理顺序是有效的

![image-20240925152112135](https://i.postimg.cc/QNfxkJzg/image-20240925152112135.png)

按热度顺序排列的处理bin使目标值下降得更快。这导致在大型问题中，当搜索时间被限制时，解的质量更高。