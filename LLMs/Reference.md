# 01. 大模型常用微调方法LORA和Ptuning的原理
LoRA这种微调的方法称为PEFT（参数高效微调）
Lora方法的核心是在大型语言模型上对指定参数增加额外的低秩矩阵，也就是在原始PLM旁边增加一个旁路，做一个降维再升维的操作。并在模型训练过程中，固定PLM的参数，只训练降维矩阵A与升维矩阵B。

Ptuning方法的核心是使用可微的virtual token替换了原来的discrete tokens，且仅加入到输入层，并使用prompt encoder（BiLSTM+MLP）对virtual token进行编码学习。

LoRA有两个主要参数，其中一个是r即矩阵降维降到的维度，另外一个是阿尔法，这个参数是对添加的旁路的梯度进行一个scale，其作用是当r维度变化的时候，梯度会发生变化，导致可能需要重新调整学习率，通过调整阿尔法就可以使得梯度变化保持同一个尺度，从而不用调整学习率。

更详细请查阅[使用 LoRA（低阶适应）微调 LLM](https://zhuanlan.zhihu.com/p/672999750)


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

而LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。

- 这里encoder是低秩，但是decoder不是的解释：由于encoder部分没有mask，因此经过softmax之后，得到的就是一个普通的矩阵，而decoder由于加上了mask，最终得到的是一个下三角或者上三角矩阵，而且这个矩阵的对角线上都是整数，也就说明，这个矩阵一定是满秩的，因此可以避免低秩问题。反过来encoder那种就不能保证一定是满秩。

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

多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。论文原作者发现这样效果确实好，更详细的解析可以查阅[Multi-head Attention](https://www.zhihu.com/question/341222779)

# 07. 监督微调SFT后LLM表现下降的原因

SFT（Supervised Fine-Tuning）是一种常见的微调技术，它通过在特定任务的标注数据上进行训练来改进模型的性能。然而，SFT可能会导致模型的泛化能力下降，这是因为模型可能过度适应于微调数据，而忽视了预训练阶段学到的知识。这种现象被称为灾难性遗忘，可以使用一些策略，如：

- 使用更小的学习率进行微调，以减少模型对预训练知识的遗忘。
- 使用正则化技术，如权重衰减或者早停，以防止模型过度适应微调数据。
- 使用Elastic Weight Consolidation（EWC）等技术，这些技术试图在微调过程中保留模型在预训练阶段学到的重要知识。

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
- 如果选择 1，2 ...类似这样数字，这些数字太大，而且在推理过程中遇到的句子长度不确定，容易导致模型泛化能力比较差。
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

# 45. 模型量化相关
- 一般模型在量化的时候，会保留最后的那个head层不变，(也就是将decoder最后的输出转换为词表尺度的那个线性层。)这样做的目的是为了尽可能的减少这里的精度损失。


# 46. RLHF技术(Reinforcement Learning from Human Feedback)基于人类反馈的强化学习
1. 首先预训练一个LLM，InstructGPT在这个基础上又进行了sft，但是由于它的数据量不大，导致一个epoch之后，直接过拟合了，但是问题不大，因为后续还要用强化学习进行优化。
2. 训练RM模型：
- 一般用预训练模型或者是sft之后的模型进行初始化（instructGPT中用的是sft之后的模型）
- 具体做法是将GPT模型最后一层的softmax去掉，然后直接接一个线性层，将模型的结果投影为一个标量，这个标量就是分数。(因为句子长度不同，所以这个线性投影具体是怎么做的呢？)
这里的解释(根据hf的trl代码)：首先将长度不同的样本统一填充到同一个长度，比如512，之后进行forward，得到最后的输出(通过一个线性层，直接将hidden_states投影到1维，即分数，维度变化：16,512,768->16,512,1)。然后根据前面的mask，取出每个样本的长度位置处的score，得到一个[16,1]的分数。
- 数据集的构建，一个prompt多个答案，然后人工对这些答案进行排序。
- 然后InstructGPT中的做法是，用一个rankLoss作为损失，一个prompt对应9个答案，然后一次将prompt和这9个答案分别传入model，之后随机取两个分数，然后损失函数的操作是最大化 高分数-低分数，相当于一次前向传播 是消耗了9个QA，然而损失计算的时候是，C(9,2)=36对损失。
![Alt](assert/reward_loss.jpg#pic_center)
3. PPO对sft之后的模型进行优化
![Alt](assert/PPO.jpg#pic_center)
- 首先将prompt扔到sft模型（policy）中，得到一对QA，然后将这对QA扔到reward model中，然后将prompt扔到原本不需要优化的sft model和policy中，求一个KL散度(目的是防止随着多轮优化后，模型的输出和原本有很大差距，导致陷入局部最优。第三部分是从原始预训练数据中取一些数据扔到policy中进行训练。 ->InstructGPT中的操作) 
- 这里强化学习和统计学习的区别就是：同一个x，当我的模型更新之后，输出的y就发生了变化，但是统计学习，无论怎样训练，同一个x对应的y永远是不变的。
![Alt](assert/ppo_loss.jpg#pic_center)
4. Anthropic/hh-rlhf数据集
- harmless-base:同一个问题，一个是希望的答案，另一个是有毒的答案。
- helpful:更希望的答案，另一个是帮助性不大的答案。
其实就是InstructGPT类似的套路，只不过一个question对应两个答案，包括reward的损失函数也是和InstructGPT中的一样。

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
    - 1.无法解决异常值的情况
    - 2.没有进行激活量化
## 2.AWQ

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