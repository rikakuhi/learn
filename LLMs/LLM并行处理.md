# 1. 数据并行(DP->DDP)
    始终只是数据实现了并行，也就是每张卡上都会拷贝一个模型。
### 1.nn.DataParallel
- 将模型拷贝到每个设备上，然后按照batch-size进行切分，每个GPU处理相同个数的样本。
- 一般有个要求是需要将model进行to('cuda:0')，使得梯度在GPU:0上完成聚合，然后将聚合的梯度广播到不同的卡上，进行参数更新。
- 多个GPU把数据传给GPU:0的时候，可能成为一个通信的瓶颈。
- 缺点：有时候GPU:0本身可能会OOM
### 2.nn.parallel.DistributedDataParallel（用nccl做通信）
- 将模型和优化器在每张卡上都拷贝一份，DDP要做的就是始终维持模型参数和优化器的状态在不同卡上都保持一致。
- 这里设置的batch-szie实际上是单张卡的batch size，每张卡反向传播之后得到梯度，各个卡之间做一个平均，相当于是bs扩大了world size倍。
- 使用torchrun启动训练，自动分配好对应的rank等参数了就。
- 通过DistributedSamper将数据分发到不同的GPU上。
- ring all-reduce-algorithm算法(环形的，避免GPU间出现的通信瓶颈)
- word：控制所有的进程
- rank：对每个进程间进行标识。一般是[0, word size - 1]
- local_rank：每个node上标记的不同GPU
- node：一台服务器
# 2. 模型并行(这里应该是流水线并行 pipeline parallel)
    这里是将不同的层放到不同的设备上
### 1.hf中device_map
- 在加载模型的时候，设置device_map= 'auto'，就已经能够实现模型并行。顺序：GPU>CPU>Disk
- 在构建模型的时候，可以直接把模型的不同层放到不同的GPU上，直接 layer. to('')即可，但是要注意在数据流转到某一层处，需要把数据也放到对应的GPU上才行。

# 3. Tensor parallel
    是一种更细粒度的分配，把每个矩阵都分到不同的设备上。在不同GPU上计算完的结果，尺度是相同的，然后把这些结果求和可以得到最终结果。
![Alt](assert/tensor_parallel.png#pic_center)
- 在计算attention的时候是在head那个维度进行拆分的。qttention的维度是[bs, num_head,seq_len, embedding]


# 4.多进程之间的通信
- 英伟达显卡无脑选用nccl(只适用于英伟达的显卡)。
- MPI：可以实现GPU、CPU之间的多设备通信，但是需要单独编译。
- GLOO：也是一种通信方式。
### 1.all reduce
多GPU之间'点对点'通信的时候，是采用python的 recv / send实现的，是阻塞的。
![Alt](assert/parallel.png#pic_center)

经过allreduce之后，所有的GPU拿到的都是T0+T1+T2+T3的和。
若有一个没有计算完，会等待其运行结束之后，才会继续执行。

### 2.nccl中集合式通信
- scatter：拆分一个tensor，放到不同的GPU上
- gather：把不同GPU上的tensor，聚合到rank=0上 (以list的形式存放)
- broadcast：将rank=0的数据复制到其他的GPU设备上
- reduce：将其他GPU上的数据加到rank=0对应的GPU中(相加)
- all-reduce： == reduce + broadcast
- all-gather： == gather + broadcast
- reduce-scatter：（每个rank中都是一个list，先相加，然后将list元素分发到不同的GPU上）

# 5.automatic mixed precison
    默认是float32和float16混合精度
    一般是在计算激活和前向传播的时候，用的是float16，但是在进行反向传播的时候，会转为float32
- 假设一个模型的参数为x，那么存储模型的参数和梯度(float16)，需要 2x + 2x bytes。如果选用Adam优化器，那么会存储momentum + variance（float32）共 4x+4x，此时计算的时候参数也用的是float32，那么也就是需要4x，也就是这样计算的话，需要 16x。
- 一些需要比较高精度计算的时候，会采用32位，比如说：求一些softmax、loss(weight unpdate)等，一般还有loss scale，就是反向传播的时候，避免很多特别小的梯度变为0，因此乘以一个比较大的数，进行扩大一下。然后再进行更新的时候，在unscale回原来的值。

# 6.deepspeed框架 
- 首先deepspeed支持 ==mpi==(跨节点通信的库，适用于集群上分布式训练)、==gloo==(高性能分布式训练框架，支持CPU或者GPU的分布式训练)、==nccl==(英伟达提供的GPU专用通信库，广泛用于GPU上的分布式训练，应该是不支持分布式训练)
- Zero将模型参数分成三部分，gradient(反向传播过程中的梯度)，模型参数，优化器参数。
- deepspeed采用的张量并行是在计算的时候，把其他卡上参数拿到同一张卡上，然后开始计算。从这个角度来说，deepspeed的计算都是在同一张卡上完成的，属于数据并行。
- 本质上其实是数据并行，但是要做好异步计算的处理。在每次计算的时候，都会先通过通信，得到完整的权重以及激活值，然后进行计算。相对比之下，megatron是使用张量并行实现的。

### 1.Zero-0
不使用切片技术，仅用DeepSpeed作为DDP。
### 2.Zero-1
- 分割Optimizer states。优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数。减少了4倍的内存，通信容量与数据并行性相同。
- 为什么通信容量与数据并行相同（没有额外的通信开销）：因为如果是单纯的数据并行，也会将所有GPU上的分别对应的数据计算完了之后，会把梯度、以及对应的状态（momentum和variance）做一次all-reduce。而如果只把优化器的状态分别存储，也是进行了这样类似的一个通信。
### 3.Zero-2
- 分割Optimizer States与Gradients。每个memory，只保留它分配到的optimizer state所对应的梯度。这很合理，因为梯度和Optimizer是紧密联系在一起的。只知道梯度，不知道Optimizer state，是没有办法优化模型参数的。
- 这个同样不会增加额外的通信开销。
### 4.Zero-3
- 分割Optimizer States、Gradients与Parameters，或者说，不同的layer. ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory。
- 这个会增加一部分通信开销，因为在计算前向传播的时候，需要把模型参数从其他GPU上拿过来，进行计算。
- 但是当层数很多的时候，也可以通过异步，来减缓这种通信开销造成的影响。（在计算前面层的时候，就提前把面层的参数拿过来）。
- 通信开销增加了原来的0.5倍。

### 5.ZeRO-R
- 是将激活值也分到不同的GPU上进行存储，(原来megatron LM 这篇论文，是在每一个GPU上都保存一个激活值，目的是减少all-reduce的次数）。
- 在计算的时候，先通过一次GPU之间的通讯，将不同GPU上的激活值拼成一个完整的输入，在进行后续的计算。
### 6.buffer
就是在发送的时候，每次都攒够一定的数据量之后，在进行发送。
### 5.ZeRO-Infinity
ZeRO-3的拓展。允许通过使用 NVMe 固态硬盘扩展 GPU 和 CPU 内存来训练大型模型。ZeRO-Infinity 需要启用 ZeRO-3。
### 6.Zero-Offload
将部分计算放到CPU和内存中进行：
- forward和backward计算量高，因此和它们相关的部分，例如 参数W（fp16），activation，就全放入GPU。
- update的部分计算量低，因此和它相关的部分，全部放入CPU中。例如 optimizer states（fp32）和gradients(fp16)等。
