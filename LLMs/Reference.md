# 01. 大模型常用微调方法LORA和Ptuning的原理
### 1.LoRA
LoRA这种微调的方法称为PEFT（参数高效微调）
Lora方法的核心是在大型语言模型上对指定参数增加额外的低秩矩阵，也就是在原始PLM旁边增加一个旁路，做一个降维再升维的操作。并在模型训练过程中，固定PLM的参数，只训练降维矩阵A与升维矩阵B。

Ptuning方法的核心是使用可微的virtual token替换了原来的discrete tokens，且仅加入到输入层，并使用prompt encoder（BiLSTM+MLP）对virtual token进行编码学习。

LoRA的两个矩阵初始化，A采用随机正态分布(均值为0，方差为西格玛)，B矩阵为0，这样设置是为了最初开始训练的时候，能够让模型按照原本预训练好的模型的初始方向进行优化。但是这样会导致A和B权重不对称的问题，因此设计了一种方法是，在训练前先在主权重中减去A*B的权重，此时可以让A和B都用正态分布的初始化。

LoRA有两个主要参数，其中一个是r即矩阵降维降到的维度，另外一个是阿尔法，这个参数是对添加的旁路的梯度进行一个scale，其作用是当r维度变化的时候，梯度会发生变化，导致可能需要重新调整学习率，通过调整阿尔法就可以使得梯度变化保持同一个尺度，从而不用调整学习率。另外一个作用是用来控制LoRA这部分的权重对网络最终的输出的影响的一个程度，阿尔法越大，对最后的结果影响也越大，反之则越小。
- 有一篇论文是说LoRA

更详细请查阅[使用 LoRA（低阶适应）微调 LLM](https://zhuanlan.zhihu.com/p/672999750)
### 2.QLoRA
#### 分位数量化
详细见：https://zhuanlan.zhihu.com/p/666234324
- 思路是将输入数据进行量化的时候，我们希望k-bit的整数值出现的频率都是相等的，因此使用分位数量化，将张量分成大小相同的若干块，这样得到更加均匀的量化特征。
- 对于4bit分位数量化，需要找到15个分位数，来将正态分布的面积平均分成16份。其中，两个分位数的中点就是对应的q
- 对于均匀量化来说，是将-1,1之间均匀的分割开，如果我们需要量化的参数是均匀分布在-1,1之间的话，能够取得一个比较小的量化误差。
- 因为训练好的权重，我们假设认为其大致满足正态分布，因此按照这种分位数的方式，进行量化在信息论的方面是最优的。
- 为了解决outliers feature的影响，进行分块处理，比如说64个参数为一组，共享一个量化常数(32bit),然后对这个常数进行二次量化，量化为FP8。进一步缩小内存开销。
- 注意上面这个量化常数其实就是每一个块中的最大值。在量化的时候，是让当前这个权重除以他所在块中的最大值，也就是量化常数，然后找到Q中最接近的数对应的索引，就是他量化后的值。
QLoRA论文解读：https://readpaper.feishu.cn/docx/CrMGdSVPKow5d1x1XQMcJioRnQe
- 第一个创新点:提出NF4数据类型
- 第二个创新点：二重量化，每64个参数共享一个量化常数(Absmax 32bit)平均下来，相当于每个参数需要占据额外的量化空间0.5bit。因此，对这个32bit的量化常数进行FP8量化。
- 创新点三：分页优化，当GPU显存不够的情况下，将一部分参数分到CPU上处理，避免出现OOM。 
- 在计算的时候，会把NF4类型转为float16之后再进行计算。

# 02. 介绍一下stable diffusion的原理

Stable Diffusion 总共包含三个主要的组件，其中每个组件都拥有一个独立的神经网络：

![Alt](assert/sd1.jpg#pic_center)

1）Clip Text 用于文本编码。
输入：文本
输出：77 个 token 嵌入向量，其中每个向量包含 768 个维度

2）UNet + Scheduler 在信息（潜）空间中逐步处理 / 扩散信息。
输入：文本嵌入和一个由噪声组成的初始多维数组（结构化的数字列表，也叫张量 tensor）。
输出：一个经过处理的信息阵列

3）自编码解码器（Autoencoder Decoder），使用处理过的信息矩阵绘制最终图像的解码器。
输入：处理过的信息矩阵，维度为（4, 64, 64）
输出：结果图像，各维度为（3，512，512）

更详细请查阅[從頭開始學習Stable Diffusion](https://chrislee0728.medium.com/%E5%BE%9E%E9%A0%AD%E9%96%8B%E5%A7%8B%E5%AD%B8%E7%BF%92stable-diffusion-%E4%B8%80%E5%80%8B%E5%88%9D%E5%AD%B8%E8%80%85%E6%8C%87%E5%8D%97-ec34d7726a6c)

更详细请查阅[十分钟理解Stable Diffusion](https://www.ithome.com/0/668/981.htm)


# 03. 为何现在的大模型大部分是Decoder only结构

大模型从模型架构上主要分为三种：Only-encoder, Only-Decoder, Encoder-Decoder三种模型架构

- Only-encoder：例如BERT，通过在大规模无标签文本上进行预训练，然后在下游任务上进行微调，具有强大的语言理解能力和表征能力。

- Only-Decoder: 例如GPT，通过在大规模无标签文本上进行预训练，然后在特定任务上进行微调，具有很强的生成能力和语言理解能力。

- Encoder-Decoder：例如T5（Text-to-Text Transfer Transformer）可以用于多种自然语言处理任务，如文本分类、机器翻译、问答等。

- 22年有一篇论文专门 https://proceedings.mlr.press/v162/wang22u/wang22u.pdf 做了实验，确实是causal decoder这种结构对于生成式的zero-shot效果最好。另外也有一些工作表明encoder-decoder的泛化能力更强。
- 为何prefix没有被大规模采用呢？若前缀很短，那也就和decoder-only没有太大区别了，而且在训练的时候，面对各种不同的问题，很难找到一个合适长度的前缀。

而LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题(苏剑林)，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。

- 这里encoder是低秩，但是decoder不是的解释：由于encoder部分没有mask，因此经过softmax之后，得到的就是一个普通的矩阵，而decoder由于加上了mask，最终得到的是一个下三角或者上三角矩阵，而且这个矩阵的对角线上都是整数，也就说明，这个矩阵一定是满秩的，因此可以避免低秩问题。反过来encoder那种就不能保证一定是满秩。

另外还有人认为预训练难度的问题，decoder only结构获取到的信息相对于其他架构更少，因此学习难度更大，当模型尺度足够大的时候，数据足够多的时候，其上限会更高。

效率问题：decoder only支持KV cache

轨迹问题：openai率先采用了这种方式取得了比较不错的效果，自然不太愿意大幅修改架构。

归纳偏置的角度：
1. 归纳偏置（Inductive bias）就是先验知识的意思，从归纳偏置的角度来理解，decoder-only加入的人类先验知识更少，训练出来的模型泛化性会比较好，但是代价就是想要训练得到好的结果就需要使用更多的参数、更多的数据和更多的算力。而像Bert这种完形填空类的模型，通过加入人的先验知识把模型设计成这种结构，使得模型在某些特定的任务上表现的很好，导致模型的泛化性就会差一些。
2. 同理，可以类比到CNN、RNN和Transformer架构上，CNN加入了局部连接、平移不变性等这些适用于图像数据的先验知识，RNN加入了门控单元、记忆单元等适用于时序数据的先验知识，Transformer则更加纯粹，并没有明确添加适用于某种任务的先验知识，所以在小规模训练数据的条件下，CNN、RNN的效果要优于Transformer，但是在大规模数据的条件下，Transformer带来的性能提升更大。
3. 像transformer、decoder-only这种添加更少先验知识的方法，让模型基于大规模的数据自主去学习有用的特征，只要数据足够多，自动学到的特征应该就会比人工设计的好，所以性能和泛化效果都会提升，前提就是基于大规模的数据。


### 1.decoder only结构的分类
- Causal Language Model：这个就是decoder，采用单向注意力 + 模型每次只预测当前的一个单词。 单项注意力是指在计算attention的时候，构建一个causal mask矩阵，上三角是-无穷，下三角是0。
- Prefix Language Model (Prefix LM)：模型的输入是一个前缀提示，模型根据提示生成后续的文本，在前缀提示部分用双向注意力机制，但是在生成文本部分用单向注意力机制。
- 读一下GLM的论文，属于prefix LM

# 04. 如何缓解 LLMs 复读机问题
##  复读机出现的原因：
- 训练数据中本身就有大量的脏数据，或者是训练数据中某些特定的句子和短语出现的频率很高。
- 训练数据缺乏多样性的表达和语境。
- LLM是自监督，这样的训练目标有可能会导致模型更倾向于输出与输入相似的语句。
## 缓解方法
- 多样性训练数据：在训练阶段，尽量使用多样性的语料库来训练模型，避免数据偏差和重复文本的问题。
- 引入噪声：在生成文本时，可以引入一些随机性或噪声，例如通过采样不同的词或短语，或者引入随机的变换操作，以增加生成文本的多样性。
- 温度参数调整：温度参数是用来控制生成文本的多样性的一个参数。通过调整温度参数的值，可以控制生成文本的独创性和多样性，从而减少复读机问题的出现。
- 后处理和过滤：对生成的文本进行后处理和过滤，去除重复的句子或短语，以提高生成文本的质量和多样性。
- Beam搜索调整：在生成文本时，可以调整Beam搜索算法的参数。Beam搜索是一种常用的生成策略，它在生成过程中维护了一个候选序列的集合。通过调整Beam大小和搜索宽度，可以控制生成文本的多样性和创造性。
- 人工干预和控制：对于关键任务或敏感场景，可以引入人工干预和控制机制，对生成的文本进行审查和筛选，确保生成结果的准确性和多样性。

更详细请查阅[大模型常见面试题解](https://blog.csdn.net/weixin_36378508/article/details/133809694
)
==beam search==：传统的是每次选择概率最大的下一个token，beam search改成了每次多保存几个，然后进行保存概率最大的k个(这里其实就是对概率取log之后相加，得到当前的句子序列的概率)
==top_k==:先找到top k概率大的几个数，然后对其进行概率的归一化，之后从这些数中进行概率采样。
==penalty factor==：会对当前输出的token，中以前已经存在的那些概率除以一个数，从而缩小其在softmax之后的概率，降低采样到这些token的可能。

# 05. 为什么transformer块使用LayerNorm而不是BatchNorm

Batch Normalization 是对这批样本的同一维度特征做归一化， Layer Normalization 是对这单个样本的所有维度特征做归一化。LN不依赖于batch的大小和输入sequence的长度，因此可以用于batchsize为1和RNN中sequence的normalize操作。

- 为什么BN在NLP中效果差
  
  - BN计算特征的均值和方差是需要在batch_size维度，而这个维度表示一个特征，比如身高、体重、肤色等，如果将BN用于NLP中，其需要对每一个单词做处理，让每一个单词是对应到了MLP中的每一个特征明显是违背直觉得；
  - BN是对单词做缩放，在NLP中，单词由词向量来表达，本质上是对词向量进行缩放。词向量是什么？是我们学习出来的参数来表示词语语义的参数，不是真实存在的。

- 为什么LayerNorm单独对一个样本的所有单词做缩放可以起到效果
  
  - layner-norm 针对每一个样本做特征的缩放。换句话讲，保留了N维度，在C/H/W维度上做缩放。
  - layner-norm 也是在对同一个特征下的元素做归一化，只不过这里不再是对应N（或者说batch size），而是对应的文本长度。

## RMSNorm和普通的LayerNorm有什么区别？
 - LayerNorm的公式可以简化为 y = a * x + b,RMSNorm首先去掉了b这一项
 - RMSNorm不在求平均值，分母上变为 平方根（所有元素的平方相加然后处以元素个数）
## RMSNorm的好处
 - 不用计算均值了，整体计算量会变小，同时省去了偏置量，等于少优化一个参数
 - 在实验中，相比LayerNorm效果基本不会变差，而且有可能效果更好。

# 06. Transformer为何使用多头注意力机制

- 多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。论文原作者发现这样效果确实好，更详细的解析可以查阅[Multi-head Attention](https://www.zhihu.com/question/341222779)
- 另外相当于是在hidden分出了几个头，这几个头的参数分别取计算注意力，防止一块计算注意力的时候，会过于对某些地方的进行关注，从而忽略了其他部分的注意力。

# 07. 监督微调SFT后LLM表现下降的原因

SFT（Supervised Fine-Tuning）是一种常见的微调技术，它通过在特定任务的标注数据上进行训练来改进模型的性能。然而，SFT可能会导致模型的泛化能力下降，这是因为模型可能过度适应于微调数据，而忽视了预训练阶段学到的知识。这种现象被称为灾难性遗忘，可以使用一些策略，如：
其实也是对微调数据的一种过拟合，因此也可以采用一些防止过拟合的策略。
- 使用更小的学习率进行微调，以减少模型对预训练知识的遗忘。
- 使用正则化技术，如权重衰减或者早停，以防止模型过度适应微调数据。
- 使用Elastic Weight Consolidation（EWC）等技术，这些技术试图在微调过程中保留模型在预训练阶段学到的重要知识。
- 通过添加一些通用数据，或者说是定期回炉，在领域数据微调之后，定期使用通用数据进行回炉训练，从而保持模型的一些通用能力。
- 知识蒸馏：用一个通用的LLM的输出指导特定领域的模型，帮助模型保持通用的知识。(具体操作是通过通用LLM输出的概率作为target，替换原来的one-hot target)。

# 08. 微调阶段样本量规模增大导致的OOM错误

全参数微调的显存需求取决于多个因素，包括模型的大小（参数数量），批次大小，序列长度，以及是否使用了混合精度训练等。对于GPT-3这样的大模型，如果想要在单个GPU上进行全参数微调，可能需要数十GB甚至上百GB的显存。

当样本量规模增大时，可能会出现OOM（Out of Memory）错误，这是因为模型需要更多的内存来存储和处理数据。为了解决这个问题，可以尝试以下方法：

- 减小批量大小：这可以减少每次训练需要处理的数据量，从而减少内存使用。
- 使用梯度累积：这种方法可以在不减小批量大小的情况下，减少内存使用。
- 使用模型并行：这种方法可以将模型的不同部分放在不同的设备上进行训练，从而减少每个设备需要的内存。

# 09. 连接文本和图像的CLIP架构简介

CLIP 把自然语言级别的抽象概念带到计算机视觉里了。确定一系列query，然后通过搜索引擎搜集图像，最后通过50万条query，搜索得到4亿个图像文本对。然后将Text Decoder从文本中提取的语义特征和Image Decoder从图像中提取的语义特征进行匹配训练。

[如何评价OpenAI最新的工作CLIP](https://www.zhihu.com/question/438649654)

# 09. Attention计算复杂度以及如何改进

- 代码中的to_qkv()函数，即用于生成q、k、v三个特征向量

![Alt](assert/attention.png#pic_center=600x400)

```python
self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
self.to_out = nn.Linear(inner_dim, dim)
```

- 在标准的Transformer中，Attention计算的时间复杂度为O(N^2)，其中N是输入序列的长度。为了降低计算复杂度，可以采用以下几种方法：
  - 使用自注意力机制，减少计算复杂度。自注意力机制不需要计算输入序列之间的交叉关系，而是计算每个输入向量与自身之间的关系，从而减少计算量。
  - 使用局部注意力机制，只计算输入序列中与当前位置相关的子序列的交互，从而降低计算复杂度。
  - 采用基于近似的方法，例如使用随机化和采样等方法来近似计算，从而降低计算复杂度。
  - 使用压缩注意力机制，通过将输入向量映射到低维空间来减少计算量，例如使用哈希注意力机制和低秩注意力机制等。

# 10. BERT用于分类任务的优点，后续改进工作有哪些？

在分类任务中，BERT的结构中包含了双向的Transformer编码器，这使得BERT能够更好地捕捉文本中的双向上下文信息，从而在文本分类任务中表现更好。BERT的后续改进工作主要包括以下方面：

- 基于BERT的预训练模型的改进，例如RoBERTa、ALBERT等；
- 通过调整BERT的架构和超参数来进一步优化模型性能，例如Electra、DeBERTa等；
- 改进BERT在特定任务上的应用方法，例如ERNIE、MT-DNN等；


# 11. 介绍transformer算法

Transformer本身是一个典型的encoder-decoder模型，Encoder端和Decoder端均有6个Block，Encoder端的Block包括两个模块，多头self-attention模块以及一个前馈神经网络模块；Decoder端的Block包括三个模块，多头self-attention模块，多头Encoder-Decoder attention交互模块，以及一个前馈神经网络模块；需要注意：Encoder端和Decoder端中的每个模块都有残差层和Layer Normalization层。
- 为什么要除以根号d：
    - 一般当d很大的时候，向量的内积会变得很大(假设Q和K都是均值为0，方差为1的变量，那么Q转置*K之后得到的均值为0，方差为d，那么不同的key与同一个query计算出来的分数可能相差很大，有的远大于0有远小于0)，对应的softmax的梯度计算出来就会非常小(如果在计算softmax的时候，一个数特别大而其他的数不是很大，就导致这些计算出来的梯度几乎为0，只有一个梯度比较大)。除以根号d之后，相当于又将结果变换到了均值为0，方差为1的正态分布，这样就能够避免长度对attention的影响
    - 如果Q的转置*K之后，在除以根号d，将均值为0，方法为d的分布变为，均值为0，方差为1的分布了，就不存在许多梯度为0的情况了。
    - 有点类似于temperature，整体除以一个比较大的数，相当于温度特别高，从而导致不确定性更强。

# 14. 在大型语言模型 (llms) 中减少幻觉的策略有哪些？

- DoLa：通过对比层解码提高大型语言模型的真实性：大型预训练 LLM 中的简单解码策略可减少幻觉;
- 在高质量数据上微调模型——在高质量数据上微调小型法学硕士模型已显示出有希望的结果，并有助于减少幻觉;
- 上下文学习：使用上下文学习为模型提供更好的上下文;
- 限制：将输出限制为受限列表，而不是自由浮动文本;

[LLMS](https://medium.com/@masteringllm/4-interview-questions-on-large-language-models-llms-1447516a8db4)


# 15. 你能否概括介绍一下 ChatGPT 的训练过程？

- 𝗣𝗿𝗲-𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴：预训练，大型语言模型在来自互联网的广泛数据集上进行训练，其中 Transformer 架构是自然语言处理的最佳选择，这里的主要目标是使模型能够预测给定文本序列中的下一个单词。此阶段使模型具备理解语言模式的能力，但尚未具备理解指令或问题的能力。

- 监督微调或者指令微调。模型将用户消息作为输入，模型通过最小化其预测与提供的响应之间的差异来学习生成响应，此阶段标志着模型从仅仅理解语言模式到理解并响应指令的转变。

- 采用人类反馈强化学习 (RHFL) 作为后续微调步骤。

# 16. 在大型语言模型 (llms) 上下文中的标记是什么？

将输入文本分解为多个片段，每一部分大约是一个单词大小的序列，我们称之为子词标记，该过程称为标记化。标记可以是单词或只是字符块。


# 18. 大模型微调的LORA原理及Lora怎么训练？

[大模型实战：使用 LoRA（低阶适应）微调 LLM](https://zhuanlan.zhihu.com/p/672999750)

# 19. lora的矩阵怎么初始化？为什么要初始化为全0？

[大模型实战：使用 LoRA（低阶适应）微调 LLM](https://zhuanlan.zhihu.com/p/672999750)

# 20. Stable Diffusion里是如何用文本来控制生成的？

Stable Diffusion是一种潜在扩散模型，主要通过自动编码器（VAE），U-Net以及文本编码器三个核心组件完成用文本来控制生成的图像。Unet的Attention模块Latent Feature和Context Embedding作为输入，将两者进行Cross Attenetion操作，将图像信息和文本信息进行了融合，整体上是一个经典的Transformer流程。

# 21. Stable Diffusion相比Diffusion主要解决的问题是什么？

Diffusion的缺点是在反向扩散过程中需要把完整尺寸的图片输入到U-Net，这使得当图片尺寸以及time step t足够大时，Diffusion会非常的慢。

# 22. Diffusion每一轮训练样本选择一个随机时间步长？

训练过程包含：每一个训练样本选择一个随机时间步长，将time step 对应的高斯噪声应用到图片中，将time step转化为对应embedding；

模型在训练过程中 loss 会逐渐降低，越到后面 loss 的变化幅度越小。如果时间步长是递增的，那么必然会使得模型过多的关注较早的时间步长（因为早期 loss 大），而忽略了较晚的时间步长信息。

# 24. 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力?

为了解决这个问题通常在领域训练的过程中加入通用数据集。那么这个比例多少比较合适呢？目前还没有一个准确的答案。主要与领域数据量有关系，当数据量没有那么多时，一般领域数据与通用数据的比例在1:5到1:10之间是比较合适的。

# 25. 在大型语言模型 (llms)中数据模态的对齐如何处理？

- Qformer

# 26. 训练通用目标检测器常会使用多源图像进行训练，如何处理新类别歧视？

- Detecting Everything in the Open World: Towards Universal Object Detection

# 27. 举例说明强化学习如何发挥作用？

一般来说，强化学习 (RL) 系统由两个主要组件组成：代理和一个环境；

![Alt](assert/Q.png#pic_center)

- 环境是智能体正在作用的设置，智能体代表强化学习算法。
- 当环境向代理发送一个状态时，强化学习过程就开始了，然后代理根据其观察结果采取行动来响应该状态。
- 反过来，环境将下一个状态和相应的奖励发送回代理。代理将使用环境返回的奖励来更新其知识，以评估其最后的行动。
- 循环继续，直到环境发送终止状态，这意味着代理已完成其所有任务。

为了更好地理解这一点，我们假设我们的智能体正在学习玩反击游戏。RL 过程可以分为以下步骤：

- RL 代理（玩家 1）从环境中收集状态 S⁰（反恐精英游戏）
- 基于状态 S⁰，RL 代理采取操作 A⁰（操作可以是任何导致结果的操作，即代理在游戏中向左或向右移动）。最初，动作是随机的
- 境现在处于新状态 S1（游戏中的新阶段）
- 强化学习代理现在从环境中获得奖励 R1。该奖励可以是额外的积分或金币
- 这个 RL 循环一直持续到 RL 代理死亡或到达目的地为止，并且它不断输出一系列状态、动作和奖励。

# 28. 如何理解强化学习中的奖励最大化？

奖励最大化是强化学习的一个关键概念。人工智能代理试图采取行动，随着时间的推移从其环境中获得最大的回报。代理从奖励或惩罚形式的输入中学习，并改变其行为方式以更好地完成工作。目标是让智能体足够聪明，能够从经验中学习并做出选择，帮助他们尽快实现长期目标。

# 29. 如何提升大语言模型的Prompt泛化性？

- 使用多个不同的prompt，从而增加模型学习的样本多样性。
- 通过在prompt中添加随机噪声或变换，来增加数据集的丰富性，从而提高模型的泛化性能。
- 采用迁移学习或元学习等方法，从先前学习的任务中提取知识，并将其应用于新的任务中。

# 30. Instruction Tuning与Prompt tuning方法的区别？

- Prompt tuning:针对每个任务，单独生成prompt模板（hard prompt or soft prompt），然后在每个任务上进行full-shot微调与评估，其中预训练模型参数是freeze的。Prompt是去激发语言模型的补全能力，比如给出上半句生成下半句、或者做完形填空，都还是像在做language model任务。（通过prompt来引导LLM完成相关的任务，并输出正确结果，并不需要sft）

[prompt tuning](https://zhuanlan.zhihu.com/p/618871247)
- Instruction Tuning：针对每个任务，单独生成instruction（hard token），通过在若干个full-shot任务上进行微调，然后在具体的任务上进行评估泛化能力（zero shot），其中预训练模型参数是unfreeze的。Instruction Tuning则是激发语言模型的理解能力，通过给出更明显的指令/指示，让模型去理解并做出正确的action。（对LLM进行sft）

[Instruction Tuning](https://zhuanlan.zhihu.com/p/558286175)

# 31. 知识蒸馏是将复杂模型的知识转移到简单模型的方法，针对知识蒸馏有哪些改进点？

- 使用不同类型的损失函数和温度参数来获得更好的知识蒸馏效果。
- 引入额外的信息来提高蒸馏的效果，例如将相似性约束添加到模型训练中。
- 将蒸馏方法与其他技术结合使用，例如使用多任务学习和迁移学习来进一步改进知识蒸馏的效果。


# 32. Transformer中的Attention计算复杂度以及如何改进？

在标准的Transformer中，attention计算的时间复杂度为O(N^2)，其中N是输入序列的长度。为了降低计算复杂度，可以采用以下几种方法：

- 使用自注意力机制，减少计算复杂度。自注意力机制不需要计算输入序列之间的交叉关系，而是计算每个输入向量与自身之间的关系，从而减少计算量。
- 使用局部注意力机制，只计算输入序列中与当前位置相关的子序列的交互，从而降低计算复杂度。
- 采用基于近似的方法，例如使用随机化和采样等方法来近似计算，从而降低计算复杂度。
- 使用压缩注意力机制，通过将输入向量映射到低维空间来减少计算量，例如使用哈希注意力机制和低秩注意力机制等。


# 33. 进行SFT操作的时候，基座模型选用Chat还是Base?

在进行SFT实验的时候，大模型选用Chat还是Base作为基座，需要根据SFT的数据量进行决定。如果你只拥有小于10k数据，建议你选用Chat模型作为基座进行微调；如果你拥有100k的数据，建议你在Base模型上进行微调。

# 34. 开源大模型进行预训练的过程中会加入书籍、论文等数据，这部分数据如何组织与处理?

进行大模型预训练时书籍和论文的文本就按照段落拆分就可以。如果进行有监督的微调任务，就需要转成指令格式的数据集，可以用标题或者关键短语作为提示。

# 35. 你能提供一些大型语言模型中对齐问题的示例吗?

一致性问题是指模型的目标和行为与人类价值观和期望的一致性程度。大语言模型，例如GPT-3，接受来自互联网的大量文本数居的训练，并且够生成类似人类的文本，但它们可能并不总是产生与人类期望或理想值一致的输出。大型语言模型中的对齐问题通常表现为
- 缺乏帮助: 当模型没有遵循用户的明确指令时
- 幻觉: 当模型编造不存在或错误的事实时。
- 缺乏可解释性: 人类很难理解模型如何得出特定决策或预测
- 生成有偏见或有毒的输出: 当受有偏见/有毒数据训练的语言模型可能会在其输出中重现该输出时，即使没有明确指示这样做。

# 36. Adaptive Softmax在大型语言模型中有何用处？

自适应Softmax在大型语言模型中非常有用，因为它可以在处理大型词汇表时进行有效的训练和推理，传统的Softmax涉及计算词汇表中每个单词的概率率，随着词汇量的增长，计算成本可能会变得昂贵。

自适应Softmax根据单词的常见程度将单词分组到簇中，从而减少了所需的计算量。这减少了计算词汇表概率分布所需的计算量。通过使用自适应softmax，可以更有效地训练和运行大型语言模型，从而实现更快的实验和开发。

# 38. 如何解决chatglm微调的灾难性遗忘问题？

https://zhuanlan.zhihu.com/p/628438318

# 40. GPT3、LLAMA的Layer Normalization 的区别是什么？

- GPT3：采用了Post-Layer Normalization（后标准化）的结构，即先进行自注意力或前馈神经网络的计算，然后进行Layer Normalization。这种结构有助于稳定训练过程，提高模型性能。

- LLAMA：采用了Pre-Layer Normalization（前标准化）的结构，即先进行Layer Normalization，然后进行自注意力或前馈神经网络的计算。这种结构有助于提高模型的泛化能力和鲁棒性。

# 41. MHA多头注意力和MQA多查询注意力以及GQA分组查询注意力的区别？
### 1.MHA(multi head attention)
- transformer中最初始的做法。
### 2.MQA(multi query attention)
- 与MHA不同的是，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头分别单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。因此在构建Linear的时候，维度应该是 所有query的维度 + 一个head的key维度 + 一个head的value维度。
- 由于多个head共用相同的key和value，这使得占用内存更少，同时访问效率更高，尽管会通过广播机制变成和MHA的维度相同，但是广播机制十分高效，并行性更强。
- 共用一个Q原理上不可行，因为MHA的设计初衷是让不同的head能够关注不同的方面，如果共用同一个Q，那多head的设计其实也就没啥用了，会影响模型捕获多样性信息的能力。
- 原论文中比较了MQA和MHA之间的精度，MQA相对于MHA几乎没有精度损失，甚至在某些方面甚至比MHA精度更高。
- 但是应该综合精度相对于MHA会降低，如果比MHA还要高，那么GQA可能就没啥用了。
### 3.GQA(group query attention)
- 相当于对MQA的改进，MQA将所有的head共用同一个k和v，这样的做法有些激进，因此假如有8个head，GQA的做法是每两个head共用一组K、V。
- GQA在实现过程中，可以任意调节gropu的数量。

### 4.关于MQA和GQA为什么可以在某些方面超过MHA
- 论文中说MHA中的多个K、V可能存在大量信息冗余。
- 减少了参数量，这可能有助于避免过拟合，提升在其他数据上的泛化能力。
- 共享参数使得多个head之间更容易协调一致，减少了训练过程中头部之间不一致导致的噪声，有助于训练出更稳定的模型。

三种attention代码如下：
```python
import torch
from torch import nn
from torch import Tensor


class Attention(nn.Module):
    def __init__(self, word_size: int = 512, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def self_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        K_T = torch.transpose(K, 0, 1)
        score = torch.matmul(Q, K_T) / torch.sqrt(self.dim_K)
        score = torch.softmax(score, dim=-1)
        Z = torch.matmul(score, V)
        return Z

    def forward(self, x: Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z


class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_head: int = 8) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.proj = nn.Parameter(torch.empty(embed_dim * n_head, embed_dim))
        nn.init.xavier_uniform_(self.proj)
        self.multihead = nn.ModuleList([
            Attention(word_size, embed_dim) for _ in range(n_head)
        ])

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.multihead], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z


class MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """

    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query: int = 8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Parameter(torch.empty(embed_dim * n_query, embed_dim))
        nn.init.xavier_normal_(self.proj)
        delattr(self, 'query')
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V) for query in self.querys
        ], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z


class GroupedQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/2305.13245.pdf
    """

    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group: int = 2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        # self.proj = nn.Parameter(torch.empty((..., ...), requires_grad=True))
        self.proj = nn.Parameter(torch.empty(embed_dim * n_grouped, embed_dim))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.grouped], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z


if __name__ == '__main__':
    attn = GroupedQueryAttention(512, 64, 4)

    attn(torch.ones(size=(10, 512)))

  ```

# 42. 推理优化技术 Flash Attention 的作用是什么？

Flash Attention 是一种高效的注意力机制实现，如共享张量核心和高效的内存使用，以减少内存占用并提高计算速度。这种方法特别适用于具有长序列和大型模型参数的场景，例如自然语言处理和推荐系统。

# 43. ZeRO，零冗余优化器的三个阶段？

- 将优化器状态分割到不同设备上，减少内存占用；除了优化器状态，还将模型参数分割到不同设备上；将梯度和优化器状态也分割到不同设备上，实现最大的内存节省。

# 44. 位置编码问题

## Absolute（transformer中的）
- 计算attention的时候，是直接计算相似度，没有考虑到不同单词的位置信息，这个信息对于理解语言也是很重要的。
- 如果选择 1，2 ...类似这样数字，这些数字太大(因为一般都是均值为0，方差为1的正态分布)，而且在推理过程中遇到的句子长度不确定，容易导致模型泛化能力比较差。
- 合格的PE应该满足如下条件
  - 为每个时间步（单词在句子中的位置）输出唯一的编码。
  - 即使句子长度不同，句子中两个时间步之间的距离应该是恒定的。
  - 模型可以轻易泛化到更加长的句子上。
  - PE必须是确定的。
- 这篇论文中为何选用：因为如果用二进制来进行表示位置，发现最低位是0和1交替，倒数第二位是0011交替，而正弦函数也有类似的周期变化的特性。sin和cos区分奇偶提取到的信息更加丰富？
- 分母上那个10000也是可以变的，具体效果可能需要做实验。
- 没有外推性，如果预训练的最大长度为512的话，那么最长能够处理的句子长度也就是512，在长就处理不了了。当然也可以将超过512维度的数值随机初始化，然后继续微调。

## Relative position embedding
- 直接体现了相对位置信号，效果也更好，具有很强的外推性，能够处理更长的文本。

## RoPE(Rotary Position Embedding)
- 提升大模型的外推性（训练句子长度较短，但是推理过程中的句子非常长的情况）
- 具有远程衰减的特性，两个token的相对位置越远，其值会缩减。即相近的token会获得更多的注意力。
- 对于使其就相当于 （对角旋转矩阵 * Q_states ）转置 * （对角旋转矩阵 * K_states）
- 对于角度的选择，底数一般默认为10000，但是一般更长的文本需要更大的底数，比如llama3采用了500000。但是一个相当大范围内的d可能最后结果都比较相似，因此具体的b的值可能没有那么明显。
- 在计算旋转矩阵的时候，LLama中的代码，是将q后半部分取负号放到前面，然后后面直接sin cos相乘都直接用加法来计算，这样就不用考虑旋转矩阵中的sin 和cos的正负了
- 远程衰减性质，当两个位置相差比较远的时候，点积的结果会比较小(随着位置变远而不断变小)
- 是在原本的Q、K矩阵中添加了绝对位置信息，然后通过二者做点积来实现相对位置的引入。
- 对向量的长度和方向有共轭性质，可以保留角度信息，从而更加保留的信息更加充分。

## Alibi位置编码 
    在attention部分的位置编码，通过attention部分的mask添加一个线性偏置来实现这一目标。
- 具体流程和T5相似，只不过在求出对应的相对位置之后，乘以一个权重，加到attention矩阵后，接着计算softmax。
![Alt](assert/alibi.png#pic_center)

## ==Alibi和RoPE精度损失问题==
- Alibi存在 i-j的运算，当这个运算超过256之后，就会出现精度损失导致数值不准确
- RoPE中存在sin(10000-2(j-1)/d),也会有精度损失


# 45. 模型量化相关
- 一般模型在量化的时候，会保留最后的那个head层不变，(也就是将decoder最后的输出转换为词表尺度的那个线性层。)这样做的目的是为了尽可能的减少这里的精度损失。


# 46. RLHF技术(Reinforcement Learning from Human Feedback)基于人类反馈的强化学习
## 1.PPO
1. 首先预训练一个LLM，InstructGPT在这个基础上又进行了sft，但是由于它的数据量不大，导致一个epoch之后，直接过拟合了，但是问题不大，因为后续还要用强化学习进行优化。
2. 训练RM模型：
- 一般用预训练模型或者是sft之后的模型进行初始化（instructGPT中用的是sft之后的模型）
- 具体做法是将GPT模型最后一层的softmax去掉，然后直接接一个线性层，将模型的结果投影为一个标量，这个标量就是分数。(因为句子长度不同，所以这个线性投影具体是怎么做的呢？)
这里的解释(根据hf的trl代码)：首先将长度不同的样本统一填充到同一个长度，比如512，之后进行forward，得到最后的输出(通过一个线性层，直接将hidden_states投影到1维，即分数，维度变化：16,512,768->16,512,1)。然后根据前面的mask，取出每个样本的长度位置处的score，得到一个[16,1]的分数。
- 数据集的构建，一个prompt多个答案，然后人工对这些答案进行排序。
- 然后InstructGPT中的做法是，用一个rankLoss作为损失，一个prompt对应9个答案，然后一次将prompt和这9个答案分别传入model，之后随机取两个分数，然后损失函数的操作是最大化 高分数-低分数，相当于一次前向传播 是消耗了9个QA，然而损失计算的时候是，C(9,2)=36对损失。
![Alt](assert/reward_loss.jpg#pic_center)
代码实现：
![Alt](assert/RankLoss.jpg#pic_center)
3. PPO对sft之后的模型进行优化
![Alt](assert/PPO.jpg#pic_center)
- 首先将prompt扔到sft模型（policy）中，得到一对QA，然后将这对QA扔到reward model中，然后将prompt扔到原本不需要优化的sft model和policy中，求一个KL散度(目的是防止随着多轮优化后，模型的输出和原本有很大差距，导致陷入局部最优。第三部分是从原始预训练数据中取一些数据扔到policy中进行训练。 ->InstructGPT中的操作) 
- 这里的kl散度有两种求法：
    - 分别计算每个对应token的kl散度，然后进行一个求和。若两个模型输出token长度并不相同，那么可以截断或者是填充。
    - 计算整个序列的概率，其实就是把所有的输出概率进行相乘得到一个整体序列的概率。这里其实暗含一个知识： 这样其实是计算的条件概率，decoder only输出的每个token概率，本质上都是一种条件概率，因为前一个token也参与了计算当前这个token的过程，所以求出来本身就是条件概率。
    另外由于计算kl散度的时候，是取了对数，那么也就不存在说，所有的概率相乘，会得到一个特别小的数，导致后续计算困难。因为乘法可以变成log中的加法。 
- 这里强化学习和统计学习的区别就是：同一个x，当我的模型更新之后，输出的y就发生了变化，但是统计学习，无论怎样训练，同一个x对应的y永远是不变的。
![Alt](assert/ppo_loss.jpg#pic_center)
- 这里损失函数中的E(x,y) ~ D表示的是在数据集D中的error。
4. Anthropic/hh-rlhf数据集
- harmless-base:同一个问题，一个是希望的答案，另一个是有毒的答案。
- helpful:更希望的答案，另一个是帮助性不大的答案。
其实就是InstructGPT类似的套路，只不过一个question对应两个答案，包括reward的损失函数也是和InstructGPT中的一样。
## 2.DPO
- 消掉了奖励模型那一项，损失函数变成了如下的形式：
![Alt](assert/DPO.png#pic_center)
- 这里的模型的输出其实就是每个token的概率相乘，然后由于取了log，所以变成了相加。
- 每次计算损失的时候，只能有一个query对应着两个answer。但是可以通过输入多个query
- 具体流程：
    - 首先已经有一个sft的模型了，首先同一个prompt让sft模型生成两个answer，人工对其进行标注排序。这样收集大量数据。其实也可以是多个回答，只不过每次计算的时候，只能算两个回答的损失。  这里的标注也可以通过reward model或者是GPT-4来标注。
    - 之后进行DPO的训练。
    - 若没有sft的模型，只有偏好数据，那么需要先在偏好数据上sft一下(用好的那个label)
# 47.Llama2中的训练策略
- 首先reward model随着收集到的数据不断增多，reward model的数据也一直在更新。（这里是训练了两个reward model：safety 和 helpfulness）
- 循环迭代：每次都让模型对需要训练的prompt产生N个答案，然后用reward model去计算分数，选择最好的那个座位标准答案。 这样就获取到了下一轮中的训练数据，即：prompt+标准答案。
- 一共迭代了5轮，前三轮的数据都是完全来自上一轮，但是会有遗忘，因此后面几轮在sft的时候就会把前面几轮中最好的数据拿过来作为训练集，这样对模型的性能有很大提升。
- 针对这个N也做了实验，N越大，可以获得更好的答案(reward model的分数会越高)。（同时温度系数也在改变，以获取更加多元化的数据）
- 在PPO优化的时候，采用了safety reward model和helpfulness reward model相结合的方式，因为事先知道一条训练数据是否可能会出发安全的红线，因此先用safety reward model计算分数，设置0.15为阈值。

# 48. CoT (chain of thought)相关
### 1.普通的CoT
- 主要是通过在prompt中添加具体的CoT步骤，或者给出一些few shot从而引导model进行合理的分析，并得出正确答案。
- 缺点：针对不同的任务，需要写不同的具体CoT以及example。

### 2.zero-shot-CoT(Let's think step by step.)
- 通过两次prompt，第一次是Q + Let's think step by step. 得到推理答案A1
- 第二次是 Q + A1 + 获取答案的prompt，不同任务不太一样(The answer is ...)得到最终结果
- 实验结果表明，这种效果最好，同时这套模版几乎使用所有的任务。
- 这些推理任务一般是不能通过Scaling Law解决的。

### 3.Manual-CoT(https://arxiv.org/abs/2201.11903)
- 用到了少样本学习，就是 在输入问题之前，手动设计一些问题和答案的样例(样例的答案给出中间的推理过程)，这些问题和答案都需要手动构造，所以叫做手动-CoT。
- 需要人工根据不同的任务，进行不同的设计，需要一定的人工成本。

### 4.Auto-CoT
- 对于每一个采样的问题拼接上“Let's think step by step”（类似于 Zero-Shot-CoT ）输入到语言模型，让语言模型生成中间推理步骤和答案，然后把这些所有采样的问题以及语言模型生成的中间推理步骤和答案全部拼接在一起，构成少样本学习的样例，最后再拼接上需要求解的问题一起输入到语言模型中进行续写。

# 49. ReAct相关
![Alt](assert/ReAct.png#pic_center)
- ReAct是通过thought和act结合起来使的LLM完成对应的任务。上图中的几种不同的方法，论文中都一起做了实验，结果表明ReAct还是效果最好。
- 优点：通过thought可以得出解决一个复杂问题的思路，同时可以得到下一步应该干什么，通过Act可以调用一些外部的API，从而补充模型所不知道的知识。
- 对比CoT系列：
    - CoT推理能力很强，但是很容易出现幻觉。
    - ReAct本身的一个问题是，在Act的时候，检索的信息是错误的，那这样最终结果一般也都是错误的。
    - ReAct本身的推理能力是弱于CoT的。
- 论文中采用的做法是，通过手动标注一些thought + act的步骤，然后引导LLM根据这些人工标注好的label，去生成更多的训练数据，减少了人工标注的成本。
- 同时设计了CoT和ReAct结合的方法，因为发现CoT可以激发出model的推理能力。
    - 当ReAct没能够得到一个具体的答案的时候，转向CoT。
    - 当采用了n次CoT，但是得到的相同答案个数没有超过n/2的，则转向ReAct。
    - 这两种方法均比单独使用CoT或者是ReAct效果要好。
- ReActfinetune后的效果是最好的。

# 50.LLM实现长文本
## 1.使用更长的文本进行预训练

## 2.利用一些其他tricks进行更长文本的拓展。

# 51.Quantization
- 量化的概念：其实是将连续值转换为一组离散值。
- 神经网络的量化：就是将一些原本是 1.23、-2.32的权重量化为1、-2（浮点数变为整型）。其实也满足上面的那种概念，只不过是将一些离散值量化的更加离散了。
### 基于k-means的量化
![Alt](assert/k-means-based-weight-quantization.png#pic_center)
![Alt](assert/k-means-based-weight-quantization-train.png#pic_center)
    - 相当于先对权重矩阵进行一个聚类，然后保存一个index矩阵，和一个codebook(这个是聚类的结果) （第一个图）
    - 在训练或者是微调的时候，也是每一类中的梯度信息进行 sum 求和，然后去更新codebook。
    - 这种方法只是存储的时候采用压缩，但是进行计算的时候都是用的float数据。

### linear quantization(零点量化zero-point quantization)
![Alt](assert/linear-quantization0.png#pic_center)
![Alt](assert/linear-quantization.png#pic_center)
![Alt](assert/linear-quantization1.png#pic_center)
![Alt](assert/linear-quantization2.png#pic_center)
    - 为AWQ的前置知识，后续具体卷积的计算等推导，具体详见linear-quantization。
### 最大绝对值量化(absolute maximum quantization)
![Alt](assert/absolute_maximum_quantization.jpg#pic_center)
详细解释见：https://zhuanlan.zhihu.com/p/627436535
- 当有一种情况的时候，会导致精度很低，就是假设原本的向量中有一个100的权重，这样计算出来之后，量化后的权重会有好几个0，因此会导致有很多的信息损失。一种简单的解决方式是，分块或者是按照列(行)来进行量化。但是随着模型参数量的不断变大，这种方式也就不好用了。
- 解决方案：(混合精度，outliers这部分采用float16来计算，其余的进行量化计算)
![Alt](assert/LLM.int8.jpg#pic_center)
将outliers和常规的分开计算，分别以float16和int8/int4来计算。
- 这些outliers分布也是有规律的，大多数都分布在所有sequence的同一个维度。
- 以6.0作为一个阈值，找出所有至少包含一个outlier的维度，然后int8和float16分开进行计算。
### 数据类型介绍
- int类型：
![Alt](assert/int.png#pic_center)
- float类型：
- float16
![Alt](assert/float32.png#pic_center)
![Alt](assert/float322.png#pic_center)

## 1.GPTQ
是对OBQ的改进，这两个方法都是训练结束后，逐层进行量化，使得量化前后损失最少，具体的量化目标函数为：
![Alt](assert/Q_function.png#pic_center)
### 1.OBD(一种剪枝方法)
- 按照剪枝每个权重后对结果的影响大小顺序，进行剪枝，同时对其他权重进行一定的补偿，使得剪枝前后结果不会发生什么改变。
- 这样相当于对每个权重计算一次剪枝(可能阈值设置，不需要所有的都进行剪枝)和其他权重的更新
- 需要计算全参数的海森矩阵
### 2.OBS(剪枝方法)
- 相对于OBD的改进是，没有忽略交叉项。
### 3.OBQ(量化)
- 认为同一行的参数之间有相关，但是不同行的互不相关，因此剪枝的时候可以一列一列进行，海森矩阵也只需要在单独每一行中计算即可。
### 4.GPTQ(量化)
- 创新点1：并不需要按照某个权重对结果影响大小顺序来进行量化，因为随着模型参数在增加，随机选择权重量化和前一种方法差距不大，因为随机选择权重量化的时候，还会对其他的权重进行更新，这就导致可能原本对结果影响很大的权重，通过更新，变得影响不是很大了。
- 创新点2：延迟部分参数更新，量化过程是一列一列的进行，当前i的更新和前i-1列有关系，和后面的没关系，因此设置一个group为128列，在这128列中的参数，每次都立即进行更新，而对128列之外的，每次先记录更新量，先保持不变，当一个group量化结束后，在统一对后面的进行更新。目的是解决花大量的时间在io的读写上，减少带宽压力。
- 创新点3：用 Cholesky 分解求海森矩阵的逆，在增强数值稳定性的同时，不再需要对海森矩阵做更新计算，进一步减少了计算量。
- 缺点：
    - 1.无法解决异常值的情况(outliers)
    - 2.没有进行激活量化
- 量化到int4 激活保持float16，因此是一种==W4A6==。
- ==在推理阶段，模型权重被动态的反量化回float16，并在该数据类型下进行实际的运算。==

## 2.AWQ
- 核心思想 1 : 模型中的所有参数对结果的影响并不是相同的，因此在量化的过程中，并不是对所有的参数进行量化，而是选择一部分保留float16的精度，其余的进行低比特量化。
    - 作者做了实验，分别保留0.1%、1%、3%的参数为float16，采用了三种选取参数的方式如下： 
        - 随机挑选
        - 基于权重分布（保留哪些绝对值较大的权重）
        - 基于激活值，这里的激活值其实就是Attention的输入，比如Q = Wq * X，其中X就是激活值，这样选的原因是，在LLM中有些激活值会比较大，一般叫做（偏移值）。而反过来，一般weight的权重是更加平稳的，因此根据激活值的分布来先选择保留哪些权重为float16
    - 作者在操作过程中选取权重的时候，并没有逐个元素进行选取，而是将激活值对每一列求绝对值的平均值，然后保留平均值最大的那一列为float16，其余进行量化。
    - 这种方法的缺点：由于保存的权重的数据类型不同，因此存储起来（混合精度的矩阵），包括后面写对应的算子，都很复杂，因此有了核心思想2。
- 核心思想 2 : 
    - 是针对步骤1中选出来的那个权重，将其乘以一个大于1的数，然后对应的让其激活值 除以这个数（可以直接写到前一个线性层中的算子中），实验显示 当这个值是2的时候精度最高。（还可以通过搜索的方式找到更好地值，实现更高的精度）
    - 这样做的原因具体可以看论文中的公式，因为在对权重进行scale的时候是分组进行计算的，但是计算△的时候，是通过权重的最大值进行计算的，因此对当前这个组的权重乘以s实际上，权重的最大值可能没变，但是对应的激活又除以了s，因此整体的误差会变小。（这里的误差来及round函数，一般范围在0-0.5之间）
    - 作者的这种做法经过实验验证有效后，于是就放弃了核心思想1中的保存一个混合精度权重矩阵的做法，取而代之的是对所有的weight进行量化，然后对于原本那些需要保留的float16的权重，乘以一个扩大因子s，其余weight的s设置为1，这样来建议量化误差。
    ![Alt](assert/awq.png#pic_center)
- 具体流程：
    - 求激活值每个通道数绝对值的平均数作为 缩放因子，然后设置了一个超参数，就是缩放因子的阿尔法次方，具体找阿尔法的方式是，从0-1之间平均取20个数，分别计算量化过后的误差，选择最小的误差作为最终的超参数的值。分组的时候，是对激活值的通道那个维度（就是每个token用多少维度的向量进行表示）进行分组，一般是128为一组。一组中通道数越少，可能模型的精度会更高，但是随之模型的尺寸也会变大。

## 3.RTN(Round-to-nearest，四舍五入到最近值)
- 量化过程快速，但是通常会带来精度损失。



# 52.Flash Attention
### 1.GPU中的结构
![Alt](assert/flashattention1.png#pic_center)
- 这是一个A100 GPU的显存结构，可以看到HBM容量很大，但带宽较低
### 2.Attention算法的计算复杂度推导
### 3.Attentin中计算方法
- 线性变化，将输入序列进行尺寸变换，得到O、K、V三个矩阵， 若每个token的embedding维度是k，那么这一步的复杂度是O(n * k * 3d)
- 计算相似度得分：通过Q与K求出相似度，得到一个n*n的相似度矩阵，计算复杂度为O(n * n * d), 另外softmax的复杂度也是O(n * n * d)。也就是说普通Attention的复杂度其实是和序列长度成平方的关系
![Alt](assert/flashattention2.png#pic_center)
- 大部分时间都花在了写入HBM和从HBM中读取数据。（类似于CPU中的寄存器和内存之间的关系）
- 这样计算的最大问题是，每次操作都需要从HBM中把数据加载到GPU的SRAM中进行计算，计算结束后又需要将结果写入HBM，因此flashattention要做的就是避免这种数据之间的来回移动。
### 4.flashattention的做法
![Alt](assert/flashattention3.png#pic_center)
- flash attention的核心是想分块计算，但是这样有一个最大的问题是softmax，需要知道当前这个元素所在的某一行的其他所有元素的情况下才可以计算，因此没有办法实现直接分块。
- softmax的分块计算：
    - 定理一：在计算softmax的时候， 是e的x-max(x)就是让每个元素x都减去他这一行中最大的数值，这样做是为了保持稳定，因为假设求(1,2,300)的softmax，这样会求一个e的300次方，这个数非常大，容易发生溢出，损失精度，如果都减去300，那这样就把最大的值限制在了e的一次方，这样计算起来精度更高。  -> 这样两种方法其实是等价的，只需要让分子分母同时乘以e的300次方，就得到和原来一样的式子了。
    - 公式如下：这个是核心，可以用 [1,2,3,4]这个向量去分组验证。
    
    ![Alt](assert/flashattention4.png#pic_center)
- 根据上面图片中的公式，M表示SRAM的最大容量。 作者将Q、K、V以及最后的输出O，进行分块，每块占SRAM容量的1/4，另外分块计算softmax的时候，两个块之间合并，需要进行储存一个中间值，作者将这个中间值放到寄存器中。
- 首先取指定大小的K、V块到SRAM中（外层循环），接着取Q、O到SRAM中(内层循环)，这样将SRAM的空间恰好占满。然后分别一次计算完这个小块的注意力以及softmax的结果，直接将结果输出到HBM中。 也就是分块计算一次性完成了从KQV->结果的输出，省略了中间多次读取、写入数据的麻烦。
- 详细解释见 https://fancyerii.github.io/2023/10/23/flashattention/
- Flash Attention的计算复杂度：假设一个块的大小为b，那一个块在计算过程中就是b * b * d的复杂度，因为有k个块，所以总的复杂度就是k * b * b * d。其中k * b = n，理想情况下，当k=b的时候，计算效率更高，为 n的1.5次方*d
- 根据以上分析，Flash attention不仅降低了计算复杂度，同时减少了多次读取和写入HBM中的时间。
# 52.补充 SPDA
- 是pytorch官方推出的一个加速attention的算子(只要把query、key、value送到这个函数中即可)，具体原理不知道。  
- 这个速度也很快，据说可能比flash attention2要快？

# 53.attention中的参数量、计算量、中间激活、KV cache
## 1.Attention的参数量
- 假设embedding的长度为h，那么Attention共有 4h**2 + 4h
- mlp一般是有两层，第一层升维度到4h，第二层降维度到h，共有8h**2 + 5h参数
- 一个Decoder模块一般有12h**2 + 13h
- 若采用绝对位置编码，不包含可训练的参数，若采用RoPE这种旋转位置编码，则会增加少量的训练参数
## 2.显存占用
![Alt](assert/Attention1.png#pic_center)
## 3.FLOPs估计
![Alt](assert/FLOPs1.png#pic_center)
![Alt](assert/FLOPs2.png#pic_center)
## 3.计算量与参数量之间的关联
![Alt](assert/para_FLOPs.png#pic_center)
## 4.中间激活值分析
- (见https://zhuanlan.zhihu.com/p/624740065)
## 5.Kv-Cache(有时又叫做增量式推理)
- 应用在推理过程中，因为LLM在推理过程中每次只推理出下一个词，然后再将目前所有推理出来的句子重新输入到LLM中，继续推理下一个词，这样会导致有大量的重复数据在进行计算，KV-Cache要做的就是把之前每一次的计算结果保留。
- 具体流程：
    - 首先将prompt进行输入到LLM中，然后再推理过程中保存model每一层的key states以及 value states，(就是input通过线性层得到的K和V)。
    - 然后完成当前这次推理之后，得到一个新的词，然后让这个新的词输入到LLM中，仅计算这个词的key states和value states以及query states，然后取出上一步保存的其他k和v，将其concat到一起，然后计算注意力。计算完了之后，更新保存的key和value，就是把concat之后的key和value进行保存。
    - 后面以此类推，直到输出停止的token为止。
    - 这样就相当于每次只计算一个词的key和value。

# 54. LLM的评价指标
### Perplexity(困惑度，简称ppl)
- 实际定义其实是用模型输出的一个句子所有token的概率进行相乘，然后放到分母上，进行开n次方(n指的是当前句子的长度)。但实际计算的时候直接用交叉熵求一个平均值，然后再取指数，也是起到类似的效果。具体是因为如果一堆特别小的数连乘，可能会有精度损失。 
- 开方是为了避免句子长度带来的影响，此时得到的是模型确信度的影响，然后取倒数，得到的就是困惑度了。
- 第一种计算方法是严格按照GPT之类LLM的定义，就是严格按照预测当前的词不能够看到后面的词，也就是在计算的时候加一个attention mask.
- 第二种计算方法，不使用attention mask，直接将LLM所支持的最大长度的输入，扔到LLM中进行推理，然后得到一个[1, max_len, hidden_states]的向量，之后通过一个head，将其转变为[1, max_len, 词表的长度]。接下来和输入input_ids计算交叉熵，注意：计算交叉熵的时候需要将整体的token后移一位，也就是将[1, i, -1] = [1, i-1, -1]，因为当前词预测的实际上是下一个词语的概率。 然后计算所有token的平均交叉熵得到数值，放到指数上进行计算。
- LLM的其实损失函数和ppl之间就差了一个取==指数==。

# 55. LLM预训练相关知识
## 1.不同预训练阶段的数据选择
一般认为预训练分为三个阶段：快速收敛阶段、稳定阶段、退火阶段（这个划分是根据Loss来判断的，一般可能会在这个阶段，略微提升一下学习率）。(退火阶段Loss下降比稳定阶段更快，可以认为这里是模型学习效率比较高的时候)
### 1.初期高质量样本
- 在训练初期就添加高质量样本，这会加速初期模型的收敛速度，随着后面模型到了平稳阶段以及退火阶段，逐步添加更多的普通样本。
- 初期加入高质量样本，由于数量比较少，少数样本多次重复学习，会导致过拟合？但人类本质就是复读机，重复的学习某个内容。
### 2.末期高质量样本
- 在退火阶段，加入高质量样本来做教科书式的学习
- 会不会影响泛化？
### 3.全程高质量样本(PHIL)
- 这个的目的就是探究小模型在某个特定领域的SOTA能力
## 2.探究喂多少token，才能将一个模型真正的充分训练
- 有一种计算每一层输入和输出cos的方式，来判断模型是否训练充分。
- 如果这个值比较小的话，有可能是每一层都在做不同的事，但是也有可能是因为句子太简单，模型对结果太肯定，导致更深层的能能力没有被挖掘出来。
## 3.batch_size问题
- batch_size比较小的时候，波动比较大，容易跳出局部最优值，从而具有更好的泛化能力。
- 比较大的时候，步数整体变少，波动也会比较小，因此有可能收敛比较慢。
## 4.现在主流的学习率调整方法
显示warm_up线性增长，然后用cosine(学习率以余弦函数的方式周期变换)
##  5.Llama3.1
- 首先在不同的评估集上面用了好多的few-shot以及CoT，有一点取巧的方式。
- 关于评估集： MMLU是一个多选题目，IFEval是一个查看模型是否按照指示操作的数据集。
## 6.post-pretrain
- 指的是在通用数据预训练结束之后，为了某个增强模型在某个专业领域中的能力(比如是法律)，需要构造专业领域的数据集，同时必须搭配上一些通用数据，从而保证模型的通用能力不至于下降的太厉害。
- 这个阶段所用的数据集，一般少于通用预训练时候的数据集，但是多于sft时候的数据集。
- 首先也要用warm up，目的是为了积攒一些动量，从而确定一个更加准确的下降方向，不至于一开始训练就走错方向，导致后期训练效果比较差。
- 数据配比，一般可以中文、英文一比一，然后一些代码以及推理性的数据不能少(这部分做过实验，对模型的能力有很大的影响)。
- 一定要画==channel loss==，根据不同的数据，构造不同的验证集，画出不同的loss曲线，这样可以更加直观的看到，在专业领域的loss下降情况。
- post pretain训练过程中，针对不同loss的变化，制定不同的策略：
    - 初始 loss 低：任务简单，或者模型已经训过这份数据。如果你使用的底座模型效果巨强，比如是 Qwen2-72B，Llama3-70B，你甚至可以断言这个数据的质量很高（能力差的小模型不能随便下定论）。当然，loss 低也有可能存在一种情况，那就是数据十分的脏，全都是重复 token 或者 固定 pattern；
    - 初始 loss 高：好现象，说明模型没有见过这个数据。但也有数据质量很差的风险，最好再清洗下这个数据源；
    - loss 持平或缓慢下降：好现象，没有比这更好的现象了，基本就是我们蒙对了底座模型 pretrain 阶段使用的数据配比才会有的现象；
    - loss 快速下降：说明这个数据很容易学习，有可能是 domain 数据的特点比较显著，也有可能是数据比较脏，都是固定 pattern 或者具有明显的格式（提一句，Llama 说任何 markdown 数据都对模型性能有损失，所以有明显格式的数据要慎重使用）；
    - common channel loss 下降明显：你的 common 数据显然不够 common，它相对模型来说有可能更像是 domain 数据，说明当前数据配比和 pretrain 的配比偏离有点远；
    - domain channel loss 下降明显：好事，鼓掌欢呼；
    - domain channel loss 不下降：初始 loss 低说明模型大概率已经训过这份 domain 数据了，初始 loss 高还不下降，可能是数据不够干净，也可能是数据比较难学，再多训会吧；
    - loss 上升：和导师或领导汇报就说学习率设置的不合适，自己私下再顺带 check 一下训练代码；

# 56. vllm原理 
### 1.当前存在的问题
- kv cache一般在内存中都是连续存储，如果处理多个请求的时候，就需要分配多个连续的长空间用于存储不同请求的kv cache。
### 2.创新点：
#### 1） PagedAttention
- 将kv cache分块存储，计算的时候当一个新的q分别和每个块里面的q*K，但是注意：softmax的时候需要将所有块的计算结果加到一起。
![Alt](assert/vllm1.png#pic_center)
![Alt](assert/vllm2.png#pic_center)
#### 2）kv cache manager
- 采用类似操作系统虚拟内存的方式，将内存划分为固定大小的页面，并将用于的程序逻辑页面映射到物理页面，允许用于使用程序访问不连续的内存空间。
- 这样就避免了每次预先都需要分配大量的固定长度的连续内存用于进行推理。
- 具体映射关系如下：
![Alt](assert/vllm3.png#pic_center)
- 填充过程中，都是从左到右进行填充，只有当一个块存储满了之后才会新开辟一个块进行存储，这样的目的是能够更高效的利用LLM。当一个request处理结束之后，会立即释放其所占用的显存，从而能够腾出空间来缓存其他的request的kv cache。
- 若一个prompt，生成多个答案，那么prompt相同的部分可以共享kv cache。
- 共享前缀，比如一般的LLM最开始输入都是system，可以将这部分的kv cache，对应可以优化质检新项目中的那个超长的prompt。
- 当请求数超过vllm的最大容量的时候，vllm采用first come first serve。
- ==交换==：当GPU物理显存被占满的时候，会将一部分请求交换到CPU显存中去。当优先级最高的那个request处理结束之后，会将其占用的显存都释放，从而将CPU内存中的那些数据写回到GPU中。也就是说，在CPU内存中存放的kv cache最大不会超过GPU本身的物理显存。
- ==分布式执行：也就是支持tensor parallel==：将Attention Head平均分到多个GPU中，因此，如果在部署过程中 注意力头数/GPU数 不能整除，则会报错。另外，对于mlp层，同样也是将线性层平均分到多个GPU上面，所以，这里也需要保证 可以整除才行。
- 支持流失输出。

# 57.RAG相关
1. 以llama-index为例
- 将外部的数据集分chunks，然后用embedding model进行处理，得到每个chunks对应的向量，然后存储起来。同时也将query进行向量话之后，和存储的向量进行相似度的计算。取出topk相关性，然后丢掉LLM中进行输出。
- llama-index在划分chunk的时候，默认每个chunk的长度是1024(具体操作是用1024-文档的路径的长度)，相邻chunk之间有重叠大小为200。
- 这里需要注意，在得到向量数据之后，query和向量数据是以特定的template输入进LLM中的(比如QA template)。
- 一个trick：可以先把query扔到LLM中得到一个答案之后，用答案去检索，这样计算出来的相似度会比直接用query靠谱。

# 58. Embedding model
- 基本都是BERT模型
### 1.bge-M3
- 可以实现不同语言的查询。
- 训练过程中，用到了无标记语料和已经标记好的语料，用于不同的训练阶段。为了实现跨语言的语义检索，采用了两个翻译数据集进行训练。
    - 长语料：从维基百科等网站中选取比较长的文章，然后随机选择段落，利用GPT对这些段落生成问题。从而构建sft数据集
#### 计算三个不同的相似度
- Dense retrieval：将query输入到类似BERT的encoder网络中，然后取最后一层中CLS对应的位置的隐藏层，将其进行norm之后，得到query的embedding。同样的对应的段落(message)也是这样得到的。两个句子之间的相似性，通过两个embedding的内积来计算。
- Lexical Retrieval(词汇检索：对长文本检索有好处)：将每个词对应的最后那层输出的隐藏层同一个线性层转换到1维，然后通过一个ReLU激活，得到一个浮点数，这个浮点数就认为是这个单词的权重，若一个词在query中多次出现，则仅保留其最大权重。将一个句子中所有的词都计算出其权重，然后用隐藏层的输出*对应的权重，之后求平均值作为文本的稀疏表征。对应passage中词的权重也这样来计算。
- multi-vector retrieval(多向量索引)：给定一个文本，得到语言模型最后一层所有位置上的隐藏状态，经过一个全连接和norm后得到文本的多向量表征(维度是n*d，n是文本长度，d是隐藏层的状态)。然后给定一个query，取query每个位置的隐藏层向量经过全连接层和一个norm，然后分别计算和文本每个位置的内积，取最大值作为得分，然后将query上所有位置跟doc的相似度求一个平均值，得到最后的一个相似度分数。
#### 训练的三个阶段
- 首先是预训练阶段，（就类似训练一个BERT）这采用的是RetroMAE的方式
![Alt](assert/RetroMAE.png#pic_center)
![Alt](assert/RetroMAE1.png#pic_center)
    - 在encoder和decoder部分采用非对称的mask(encoder部分用15%-30%，decoder部分用50%-70%)
    - 在A和B阶段采用交叉熵计算损失。由于decoder的结构十分简单，因此解码过程很有挑战性，可以迫使生成更高质量的sentence embedding。
    - 增强解码(作者认为交叉熵只能利用相同的上下文来进行计算，也就是说每次都只用了固定位置的token的隐藏层)：
    ![Alt](assert/RetroMAE3.png#pic_center)
    ![Alt](assert/RetroMAE4.png#pic_center)
    - 随机抽样
    ![Alt](assert/RetroMAE5.png#pic_center)
- 第二阶段：
    只考虑dense部分的对比学习损失(InfoNCE)拉近query和text之间的距离，同时疏远query和不相关文档之间的距离。
- 第三阶段：使用了一种自激励蒸馏方法来提高检索性能，具体的是计算出那三个分数之后，对这三个分数进行一个取平均，然后作为target，让这三个分数分别取学习这个target，从而提高单检索模式的效果。这个第三阶段应该 是在第二阶段的基础上，进行的，就是第二阶段的infoNCE损失仍然计算(这里是计算了三个infoNCE)。
第三阶段所用的蒸馏损失：
![Alt](assert/RetroMAE6.png#pic_center)
#### 其他的一些tricks
1. 长文本的优化：
    ![Alt](assert/RetroMAE7.png#pic_center)
2. 训练效率的优化：
    ![Alt](assert/RetroMAE8.png#pic_center)

#### 评估指标
一般就是在一些事先指定好的数据集上，然后求召回率，或者是说求Top-K的召回率。

# 59.PreNorm与PostNorm的区别
1.PostNorm在残差之后做Norm，对参数的正则化效果更好，进而导致模型的鲁棒性要好。
- 容易证明，如果x的方差（二阶矩同理）为σ1的平方，而F(x)的方差为σ2的平方，x + F(x)的残差会变成 σ1的平方 + σ2的平方，也就说明方差会进一步被放大，如果此时后面加一个norm，会缩小方差。但是却会缩减残差连接的作用。

2.PreNorm有一部分参数直接加到了后面，不需要对其进行正则化，也就是相当于前面的信息可以直接传递到后面，从而在训练过程中更容易优化。但是存在一个问题就是会很依赖残差层，相当于整体网络的深度变浅了。
3.苏剑林：post-norm训练出来的效果要更好，pre-norm更容易训练，现在大多数主流的LLM用的大多数是pre-norm。
4.还有就是能不能把norm放到attention的后面？答案是不太合理：因为这样导致经过attention之后的数值由于有norm，导致整体数值比较小，但是残差那一层没有经过norm，可能数值比较大，从而导致残差的信息占主导了，这样貌似不是很合理。(但是有一篇定会就是这样做的....)


# 60.Toolformer(LLM可以调用各种API)


# 61.Dual Chunk Attention
将比较长的上下文进行分块处理，先分别计算每个块里面的注意力机制，然后再将每个块中所有token进行的feature计算一个平均值，之后计算所有块之间的相似度。

# 62.prefix tuning

# 63. p tuning

# 64. MOE模型
- 主要是将Transformer的FFN部分替换为MOE层，其中MOE包括一个门控网络和若干个专家。
- 门控：其实就是将attention之后的hidden，做一个降维，降低到专家的个数(qwen
2中的54B那个是有64个专家)，之后取softmax之后得到概率，然后通过top_k取出需要的专家(qwen2采用8)。为什么不是每次都选最优的专家呢？答案：因为在训练过程中是想让门控学习如何更有效的选择专家，所以一般不会单独只有一个专家。
- 在训练过程中，如果不做特殊的处理，可能会聚焦几个特定的专家系统，从而使的网络更快的收敛，因此这里，需要设计一个辅助函数。
- 辅助函数：
![Alt](assert/MOE.png#pic_center)
![Alt](assert/MOE1.png#pic_center)
    - 最后将这个辅助函数加到最后的loss中。
    - fi是一个batch中，被分到第i个export的数量。
    - bi是把分到每个expert中的概率进行相加了，这里是可微的。
    - 具体代码实现中，其实是对概率进行先取指数，之后求和，然后再求对数。目的是直接对这些值取对数是不稳定的，防止数值计算中的上溢或者下溢。
    - 之后对这个进行平方，然后对整个批次中所有位置求平均值。
```
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    计算负载平衡损失函数。

    负载平衡损失函数用于Switch Transformers中，旨在通过调整专家分配概率来实现负载平衡，以优化模型性能。

    Args:
        router_probs (torch.Tensor):
            形状为 [batch_size, sequence_length, num_experts] 的输入概率张量，表示每个位置选择每个专家的概率。
        expert_indices (torch.Tensor):
            形状为 [batch_size, sequence_length] 的整数张量，表示每个位置选择的专家索引。

    Returns:
        float:
            标量，表示计算得到的负载平衡损失值。
    """
    num_groups, tokens_per_group, _ = router_probs.shape
    # 计算对数概率的和，对应于每个位置的专家选择概率
    log_z = torch.logsumexp(router_probs, dim=-1)
    # 计算负载平衡损失，以提高模型的稳定性
    balancing_loss = log_z**2
    # 返回平均负载平衡损失
    return torch.sum(balancing_loss) / (num_groups * tokens_per_group) 这个函数 的作用
```

- 在推理或者训练的时候，MOE的共享层复制在所有的GPU上，而不同的专家放到不同的GPU上。

# 65. Llama3.1
### 1.预训练
- 通过去重方法和数据清理机制，去提升抓取到的数据质量，去掉一些有个人信息的网站以及成人网站的信息。论文中说如果数据中有大量的 PII数据（personally identifiable information），则直接去掉对应的数据。（可能是计算一个比例，然后如果PLL信息超过了这个比例，则直接去掉）。
- 在抓取数据的过程中，设置了一些特别的方法，更好的提取html中正文的文字，而尽量减少一些旁边栏目的影响，对于一个html中的代码块，会保留其原本的一些换行符等信息，对于数学公式会取一些art属性中的信息。
- 数据去重：1）按照url进行去重。    2）按照miniHash    3）在三千万条数据中，如果有一行出现超过6次，就去掉（可能是一些导航栏之类的信息）。
- 计算不同文章间的分布，如果有一个文档的分布和其他分布相差太大，则去掉。
- 根据DeepSpeek中一系列论文，如果训练数据中的代码和推理性的数据很多，那么模型这方面的能力相对也会很强，因此这块需要对抓取到的一些html进行一些特殊的处理，把这些数据保留下来。
- 多语言：先通过一个分类器，将不同种类的语言抓取出来，然后进行去重。同时也会去用之前的一些模型，对每种不同的语言进行数据的打分，找到更多的高质量的数据。
- 通过实验进行不同种类的数据的一个混合，最终是50%是通用知识，25%是数学，17%是code，8%是多语言。
# 66.分词相关知识

- 将文本按照最小粒度来拆分的缺点（中文是单个字，英文是单个字母）：
    - 对于英文来说，单个字母没有含义。
    - 难以学到词汇的真正含义，尤其对于英文来说。
    - 会让输入变得很长，训练和推理特别慢。
- Subword:
    - 有一定的泛化能力，即使没有遇到过的词，也会通过subword来理解。
    - 对英文支持好，但是对中文的效果并不好。

- qwen2的词表看起来那么奇怪的原因是，首先将每个token转为对应的unicode码点(类似ASCIll)，之后进行一定调整，将其范围限制在0-255上，然后用bytes处理，转为二进制，再用uft-8编码，得到最后的中文。
```python
def analyze_string_decode_to_chinese(input_string):
    print("原始字符串:", input_string)

    unicode_points = [ord(char) for char in input_string]
    print("Unicode 码点:", unicode_points)

    decoded_points = [point - 162 if point > 255 else point for point in unicode_points]
    print("调整后的码点:", decoded_points)

    try:
        # 将调整后的码点转换为字节
        byte_array = bytes(decoded_points)
        # 直接使用 UTF-8 解码转换为字符串
        decoded_string = byte_array.decode('utf-8')
        print("解码后的字符串:", decoded_string)
    except Exception as e:
        print("解码失败:", e)

```
### 1.BPE
    其实就是找到一些词具有的共同前缀，比如说 love loved lovest这三个词之间有一定的关系，如果分别单独作为一个token，则不能体现出其之间的关系。
- 核心是从字母开始(汉字则是从单个汉字好开始)，不断找词频最高、且连续的两个token合并，直到达到目标词数。
- ==优点==：可以缓解没有见过的生词问题，减少了词表的大小。
- ==缺点==：
    - 基于频率统计的，对于语料有很大的依赖性，如果语料很大，使用这个可能会效果很好，但是如果语料很小，结果可能差强人意。
    - 如果遇到了unicode编码的话，有可能导致词表很大。
    - 有可能在decode部分出现歧义

### 2.Byte-level BPE（Qwen2用的这种）
- 为了解决unicode导致BPE得到的词表过于大的情况，选择使用byte(字节)作为单位进行BPE的操作。因为有些中文的unicode编码可能是由2-4个字节来表示的。
- 这个是需要先将不同的语言都进行分词(中文用一个 /w 来分词或者是用单个词)，之后在进行聚合。
- 解决了BPE中可能会有单个生僻字符，单独占领一个token的情况，因为一般一个字符是由1-4个字节构成的。
- 可以缩小词表的大小。
- 同时在多语言中能够更好的共享，因为所有的语言都可以用unicode编码。

### 3.WordPiece
- 其思想和BPE一样，只不过在合并的时候，并不是按照出现最大频数进行合并的，而是通过条件概率，每次都保证下面这个式子最大：
![Alt](assert/wordpiece.png#pic_center)
- 优点：能够合成相对于更有意义的子词。
- 这里的概率也可以直接用频数来计算，因为频数和概率其实就是多除以一个总数的问题。
### 4.sentence piece
- 将一个句子看成一个整体，在拆分成片段，没有保留天然的词语的概念，将空格也看成一种特殊字符来对待，然后进行BPE。
- 就是省去了BPE那种首先通过一些工具进行分词(中文用jieba，英文用空格)，然后再聚合的形式。
# 67. Tokenizer的作用
- 分词
- 将词转换为input_ids，以及从input_ids转为正常的文字
- 填充到同一个长度
- 支持对话模版，将对话转为一些模版。
- attention_mask，比如补齐的时候，就需要添加上attention mask
- 可能会在词表中添加一些不同模型的特殊token
- 截断，若超出最大长度，则对齐进行截断
- 特殊符号处理（比如EOS、PAD等）
- 在词表中添加一些新词。
- 支持tensor parallel，即对输入进行分块，方便送到不同的GPU中。
- 可以训练，这里的训练是指利用BPE等算法，重新生成符合自己数据的一个词表。
    - 训练的时候，需要先构建一个基本的词表，这个词表可以是有些模型的tokenizer，应该也可以从头构建(没试过)。

# 68. 微调的时候的一些参数设置
- 一般sft的时候的学习率要设置到预训练的0.1，比如pre-train的学习率是5e-4，那么sft的学习率就是5e-5

# 69.T5的相关知识
### 1.位置编码
- 首先计算q和k之间的一个相对位置，其实就是位置索引进行相减，得到相对位置，这里有一个机制，就是在差值在一定范围之间的时候，采用同一个距离，即j-i对应的真是位置实际上是f(j-i)，这里做了一个映射。然后构造一个nn.embedding，取对应位置的embedding加到注意力矩阵上。之后进行softmax
- 缺点：
    - 1.这些参数都需要训练。   
    - 2.外推性比较差

# 70.qwen2相对于之前的一些改进
- causal attention：那些为1的需要设置为0，然后mask中为0的设置为 -无穷，之后加到attention的矩阵上。
### 1.RoPE
### 2.RMSNorm
### 3.SwiGLU
### 4.GQA
### 5.另外在预训练的时候采用了dual chunk attention等技术
### 6.QKV bias for attention
### 7.预训练过程中，扩充了高质量代码、数学以及多语言数据。(7T tokens)
### 8.扩充到12T后，效果没有什么提升
### 9.为了增强长上下文能力，预训练的最后阶段将长度从4096增加到32k，并增大旋转位置编码的底数。
### 10.slide window
    在生成文本的时候，只和window范围内进行attention计算。
    是否会造成信息损失？相当于没有依赖当前token之前的全部token？
    答：由于当前window内的token生成，都是依赖于之前的window生成的。也就是说其实也包含之前的一些信息。