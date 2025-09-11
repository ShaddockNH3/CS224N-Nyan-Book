https://web.stanford.edu/class/cs224n/assignments_w25/a3.pdf

### **第一部分：神经机器翻译 (NMT) 

> In Machine Translation, our goal is to convert a sentence from the source language (e.g. Mandarin Chinese) to the target language (e.g. English).

在机器翻译这个魔法领域里，我们的目标，是把一句话从一种语言（比如中文），**变成另一种语言**（比如英文）。

> In this assignment, we will implement a sequence-to-sequence (Seq2Seq) network with attention, to build a Neural Machine Translation (NMT) system.

在这次的任务中，我们将要亲手实现一个带有“注意力机制”的“序列到序列”（Seq2Seq）网络，来建造一个我们自己的“神经机器翻译”（NMT）系统。

> In this section, we describe the training procedure for the proposed NMT system, which uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.

在这一节里，我们就来详细描述一下我们设计的这个 NMT 系统的“训练流程”。它主要由两个核心部件构成：一个 **“双向长短期记忆”编码器** 和一个 **“单向长短期记忆”解码器**。

#### **模型描述 (训练流程) - Model description (training procedure)**

> Given a sentence in the source language, we look up the character or word embeddings from an embeddings matrix, yielding x1, . . . , xm (xi ∈ Re×1), where m is the length of the source sentence and e is the embedding size.

当我们拿到一个源语言的句子（比如一句中文）时，第一步，我们要从一个叫做“词向量矩阵”的“大字典”里，查出句子中每一个字或词对应的“数字身份证”，也就是词向量。这样，我们就得到了一串向量 $x_1, \dots, x_m$。这里的 $m$ 是源语言句子的长度，而 $e$ 是每个词向量的维度大小。

  $$
  x_i \in \mathbb{R}^{e \times 1}
  $$

> We then feed the embeddings to a convolutional layer while maintaining their shapes.

紧接着，我们会把这些词向量喂给一个“卷积层”(CNN)。这个小小的探测器非常擅长捕捉像“电+脑→电脑”这样由相邻几个字组成的“局部短语风味”，同时保持它们的形状不变。

> We feed the convolutional layer outputs to the bidirectional encoder, yielding hidden states and cell states for both the forwards (→) and backwards (←) LSTMs.

经过“风味探测器”处理后，我们再把结果喂给我们强大的“双向编码器”。这个编码器会非常严谨地，**从头到尾**读一遍句子，再**从尾到头**倒着读一遍，并为这两种读取方式分别产出“隐藏状态”（短期记忆）和“细胞状态”（长期记忆）。

> The forwards and backwards versions are concatenated to give hidden states h\_enc\_i and cell states c\_enc\_i:

最后，对于句子中的每一个词，我们会把“正向读取”和“反向读取”得到的两种记忆拼接在一起，形成一份对这个词的、无比全面的“阅读笔记”，也就是最终的编码器隐藏状态 $h_{\text{enc}_i}$ 和细胞状态 $c_{\text{enc}_i}$：

> **公式 (1):** $h_{\text{enc}_i} = [\overrightarrow{h_{\text{enc}_i}}; \overleftarrow{h_{\text{enc}_i}}]$ where $h_{\text{enc}_i} \in \mathbb{R}^{2h \times 1}, \overrightarrow{h_{\text{enc}_i}}, \overleftarrow{h_{\text{enc}_i}} \in \mathbb{R}^{h \times 1}, 1 \le i \le m$

这是在说，第 $i$ 个词的最终**隐藏状态** $h_{\text{enc}_i}$，是由正向读取到第 $i$ 个词的隐藏状态 $\overrightarrow{h_{\text{enc}_i}}$（维度为 $h$）和反向读取到第 $i$ 个词的隐藏状态 $\overleftarrow{h_{\text{enc}_i}}$（维度也为 $h$）上下拼接而成的。所以，最终的维度就变成了 $2h$ 了。

$$
h_{\text{enc}_i} = \begin{bmatrix} \overrightarrow{h_{\text{enc}_i}} \\ \overleftarrow{h_{\text{enc}_i}} \end{bmatrix} \quad \text{其中} \quad h_{\text{enc}_i} \in \mathbb{R}^{2h \times 1}, \overrightarrow{h_{\text{enc}_i}}, \overleftarrow{h_{\text{enc}_i}} \in \mathbb{R}^{h \times 1}, 1 \le i \le m
$$

> **公式 (2):** $c_{\text{enc}_i} = [\overrightarrow{c_{\text{enc}_i}}; \overleftarrow{c_{\text{enc}_i}}]$ where $c_{\text{enc}_i} \in \mathbb{R}^{2h \times 1}, \overrightarrow{c_{\text{enc}_i}}, \overleftarrow{c_{\text{enc}_i}} \in \mathbb{R}^{h \times 1}, 1 \le i \le m$

同样地，第 $i$ 个词的最终**细胞状态** $c_{\text{enc}_i}$ 也是由正向和反向的细胞状态拼接而成的。

$$
c_{\text{enc}_i} = \begin{bmatrix} \overrightarrow{c_{\text{enc}_i}} \\ \overleftarrow{c_{\text{enc}_i}} \end{bmatrix} \quad \text{其中} \quad c_{\text{enc}_i} \in \mathbb{R}^{2h \times 1}, \overrightarrow{c_{\text{enc}_i}}, \overleftarrow{c_{\text{enc}_i}} \in \mathbb{R}^{h \times 1}, 1 \le i \le m
$$

> We then initialize the decoder’s first hidden state h\_dec\_0 and cell state c\_dec\_0 with a linear projection of the encoder’s final hidden state and final cell state.

“阅读家”（编码器）的工作完成了，现在轮到“创作家”（解码器）登场了。为了给创作家一个初始的“灵感”，我们会用一个“线性投影”（可以理解成一次聪明的变换），把编码器**最终的**隐藏状态和细胞状态，作为解码器的**初始**隐藏状态 $h_{\text{dec}_0}$ 和细胞状态 $c_{\text{dec}_0}$。
* (小提示哦：这里的“最终状态”，就是**正向读到最后一个词的状态**和**反向读到第一个词的状态**的拼接，它浓缩了对整个句子的理解！)

> **公式 (3):** $h_{\text{dec}_0} = W_h[\overleftarrow{h_{\text{enc}_1}}; \overrightarrow{h_{\text{enc}_m}}]$ where $h_{\text{dec}_0} \in \mathbb{R}^{h \times 1}, W_h \in \mathbb{R}^{h \times 2h}$

我们把反向读取的第一个词的隐藏状态 $\overleftarrow{h_{\text{enc}_1}}$ 和正向读取的最后一个词的隐藏状态 $\overrightarrow{h_{\text{enc}_m}}$ 拼接起来（得到一个 $2h$ 维的向量），然后用一个“魔法转换矩阵” $W_h$ (维度是 $h \times 2h$) 把它“压缩”成一个 $h$ 维的向量，作为解码器的初始隐藏状态 $h_{\text{dec}_0}$。

$$
h_{\text{dec}_0} = W_h \begin{bmatrix} \overleftarrow{h_{\text{enc}_1}} \\ \overrightarrow{h_{\text{enc}_m}} \end{bmatrix} \quad \text{其中} \quad h_{\text{dec}_0} \in \mathbb{R}^{h \times 1}, W_h \in \mathbb{R}^{h \times 2h}
$$

> **公式 (4):** $c_{\text{dec}_0} = W_c[\overleftarrow{c_{\text{enc}_1}}; \overrightarrow{c_{\text{enc}_m}}]$ where $c_{\text{dec}_0} \in \mathbb{R}^{h \times 1}, W_c \in \mathbb{R}^{h \times 2h}$

初始细胞状态 $c_{\text{dec}_0}$ 的计算方法和上面一模一样，只是用了另一个“魔法转换矩阵” $W_c$。

$$
c_{\text{dec}_0} = W_c \begin{bmatrix} \overleftarrow{c_{\text{enc}_1}} \\ \overrightarrow{c_{\text{enc}_m}} \end{bmatrix} \quad \text{其中} \quad c_{\text{dec}_0} \in \mathbb{R}^{h \times 1}, W_c \in \mathbb{R}^{h \times 2h}
$$

> With the decoder initialized, we must now feed it a target sentence. On the t-th step, we look up the embedding for the t-th subword, yt ∈ Re×1. We then concatenate yt with the combined-output vector ot−1 ∈ Rh×1 from the previous timestep (we will explain what this is later down this page!) to produce y\_bar\_t ∈ R(e+h)×1. Note that for the first target subword (i.e. the start token) o0 is a zero-vector. We then feed y\_bar\_t as input to the decoder.

解码器有了初始灵感后，我们就要开始教它翻译啦！在第 $t$ 步：

  1. 我们拿出目标句子（英文）中第 $t$ 个词的词向量 $y_t$。
  2. 我们把它和上一步 ($t-1$) 的一个叫做“综合输出向量”的 $o_{t-1}$（这个我们稍后会解释哦）拼接起来。
  3. 这样就得到了一个新的、信息更丰富的输入向量 $\bar{y}_t$，它的维度是 $e+h$。
  4. (对于第一个词，我们还没有 $o_0$，所以就用一个全是0的向量来代替)。
  5. 最后，我们把这个 $\bar{y}_t$ 作为解码器这一步的输入。

> **公式 (5), (6):** $h_{\text{dec}_t}, c_{\text{dec}_t} = \text{Decoder}(\bar{y}_t, h_{\text{dec}_{t-1}}, c_{\text{dec}_{t-1}})$ where $h_{\text{dec}_t} \in \mathbb{R}^{h \times 1}, c_{\text{dec}_t} \in \mathbb{R}^{h \times 1}$

解码器（一个LSTM单元）接收了新的输入 $\bar{y}_t$，以及它自己上一时刻的记忆 $h_{\text{dec}_{t-1}}, c_{\text{dec}_{t-1}}$，然后进行一次“思考”，更新自己的记忆，得到这一时刻新的隐藏状态 $h_{\text{dec}_t}$ 和细胞状态 $c_{\text{dec}_t}$。

$$
h_{\text{dec}_t}, c_{\text{dec}_t} = \text{Decoder}(\bar{y}_t, h_{\text{dec}_{t-1}}, c_{\text{dec}_{t-1}}) \quad \text{其中} \quad h_{\text{dec}_t} \in \mathbb{R}^{h \times 1}, c_{\text{dec}_t} \in \mathbb{R}^{h \times 1}
$$

> We then use h\_dec\_t to compute multiplicative attention over h\_enc\_1, . . . , h\_enc\_m:

现在！魔法中最神奇的部分来了，我们用解码器刚刚得到的、最新的思考成果 $h_{\text{dec}_t}$，去对编码器写下的**所有**“阅读笔记” $h_{\text{enc}_1}, \dots, h_{\text{enc}_m}$ 进行一次“乘法注意力”计算。这就像“创作家”在下笔前，回头去“划重点”一样！

> **公式 (7):** $e_{t,i} = (h_{\text{dec}_t})^T W_{\text{attProj}} h_{\text{enc}_i}$ where $e_t \in \mathbb{R}^{m \times 1}, W_{\text{attProj}} \in \mathbb{R}^{h \times 2h}, 1 \le i \le m$

为了决定原文中哪个词最值得“关注”，我们用解码器的当前状态 $h_{\text{dec}_t}$（作为“提问”），去和每一个原文单词的笔记 $h_{\text{enc}_i}$（作为“待查资料”）计算一个“相关性分数” $e_{t,i}$。$W_{\text{attProj}}$ 是一个学习到的“投影矩阵”，帮助“提问”和“资料”在同一个频道上对话。

$$
e_{t,i} = (h_{\text{dec}_t})^T W_{\text{attProj}} h_{\text{enc}_i} \quad \text{其中} \quad e_t \in \mathbb{R}^{m \times 1}, W_{\text{attProj}} \in \mathbb{R}^{h \times 2h}, 1 \le i \le m
$$

> **公式 (8):** $\alpha_t = \text{softmax}(e_t)$ where $\alpha_t \in \mathbb{R}^{m \times 1}$

我们把刚刚算出的所有 $m$ 个“相关性分数” $e_t$ 丢进一个 `softmax` 函数里，把它变成一堆加起来等于1的“注意力权重” $\alpha_t$。这就像把分数变成了“荧光笔的墨水浓度”，分数越高的词，墨水越浓！

$$
\alpha_t = \text{softmax}(e_t) \quad \text{其中} \quad \alpha_t \in \mathbb{R}^{m \times 1}
$$

> **公式 (9):** $a_t = \sum_{i=1}^{m} \alpha_{t,i} h_{\text{enc}_i}$ where $a_t \in \mathbb{R}^{2h \times 1}$

然后，我们用这些“墨水浓度” $\alpha_{t,i}$ 作为权重，对原文所有的“阅读笔记” $h_{\text{enc}_i}$ 进行一次“加权求和”。这样得到的“注意力输出”向量 $a_t$，就是一份专门为翻译当前这个词而“划出的重点摘要”！

$$
a_t = \sum_{i=1}^{m} \alpha_{t,i} h_{\text{enc}_i} \quad \text{其中} \quad a_t \in \mathbb{R}^{2h \times 1}
$$

> We now concatenate the attention output at with the decoder hidden state h\_dec\_t and pass this through a linear layer, tanh, and dropout to attain the combined-output vector ot.

现在，我们把“划好的重点” $a_t$ 和解码器自己的“当前思考” $h_{\text{dec}_t}$ 拼接起来，然后让这个信息更丰富的向量，依次通过一个线性层（变换维度）、一个 `tanh` 激活函数（增加非线性）和一个 `dropout` 层（防止死记硬背），最终得到我们心心念念的那个“综合输出向量” $o_t$！

> **公式 (10):** $u_t = [a_t; h_{\text{dec}_t}]$ where $u_t \in \mathbb{R}^{3h \times 1}$

这是第一步，把“重点摘要” $a_t$（$2h$维）和“当前思考” $h_{\text{dec}_t}$（$h$维）拼在一起，得到一个 $3h$ 维的向量 $u_t$。

$$
u_t = \begin{bmatrix} a_t \\ h_{\text{dec}_t} \end{bmatrix} \quad \text{其中} \quad u_t \in \mathbb{R}^{3h \times 1}
$$

> **公式 (11):** $v_t = W_u u_t$ where $v_t \in \mathbb{R}^{h \times 1}, W_u \in \mathbb{R}^{h \times 3h}$

用一个“融合矩阵” $W_u$ 把 $3h$ 维的 $u_t$ 重新融合、提炼成一个 $h$ 维的向量 $v_t$。

$$
v_t = W_u u_t \quad \text{其中} \quad v_t \in \mathbb{R}^{h \times 1}, W_u \in \mathbb{R}^{h \times 3h}
$$

> **公式 (12):** $o_t = \text{dropout}(\tanh(v_t))$ where $o_t \in \mathbb{R}^{h \times 1}$

对提炼后的 $v_t$ 进行 `tanh` 激活和 `dropout` 处理，得到最终的、将用于预测和反馈给下一步的“综合输出向量” $o_t$。

$$
o_t = \text{dropout}(\tanh(v_t)) \quad \text{其中} \quad o_t \in \mathbb{R}^{h \times 1}
$$

> Then, we produce a probability distribution Pt over target subwords at the t-th timestep:

最后一步！我们要根据这个高度浓缩了各种信息的 $o_t$，来预测在这一步最应该生成的英文单词是哪一个！

> **公式 (13):** $P_t = \text{softmax}(W_{\text{vocab}} o_t)$ where $P_t \in \mathbb{R}^{V_t \times 1}, W_{\text{vocab}} \in \mathbb{R}^{V_t \times h}$

我们用一个超大的“词典投影矩阵” $W_{\text{vocab}}$，把 $h$ 维的 $o_t$ 映射成一个长度为整个目标词典大小（$V_t$）的向量。这个向量的每一位，都代表着生成对应单词的“分数”。再通过一个 `softmax`，这些“分数”就变成了“概率” $P_t$！概率最高的那个，就是我们的最佳选择！

$$
P_t = \text{softmax}(W_{\text{vocab}} o_t) \quad \text{其中} \quad P_t \in \mathbb{R}^{V_t \times 1}, W_{\text{vocab}} \in \mathbb{R}^{V_t \times h}
$$

> Finally, to train the network we then compute the cross entropy loss between Pt and gt, where gt is the one-hot vector of the target subword at timestep t:

最后，为了训练我们的网络，我们会计算模型给出的“概率分布” $P_t$ 和“标准答案”（也就是正确的那个词，用 one-hot 向量 $g_t$ 表示）之间的“交叉熵损失”。这个损失值，衡量了我们的模型有多“惊讶”。

> **公式 (14):** $J_t(\theta) = \text{CrossEntropy}(P_t, g_t)$

这就是第 $t$ 步的损失函数 $J_t(\theta)$。我们的目标，就是通过调整模型的所有参数 $\theta$ (也就是我们前面看到的各种 $W$ 矩阵)，来让这个“惊讶值”变得越小越好！

$$
J_t(\theta) = \text{CrossEntropy}(P_t, g_t)
$$

> Now that we have described the model, let’s try implementing it for Mandarin Chinese to English translation!

 现在我们已经把整个模型都描述清楚了，让我们试着为中译英任务把它实现出来吧！
 
### **编程实现**

> Implementation and written questions

#### **(a) (2 分) (编程题)**

> In order to apply tensor operations, we must ensure that the sentences in a given batch are of the same length. Thus, we must identify the longest sentence in a batch and pad others to be the same length. Implement the `pad_sents` function in `utils.py`, which shall produce these padded sentences.

 为了能让我们的魔法（张量操作）顺利施展，我们必须保证同一批次里的所有句子，都有着完全相同的长度。所以，这一步需要你找到一批句子中最长的那一句，然后用特殊的“填充”符号，把其他所有句子都补到和它一样长。请在 `utils.py` 文件中实现 `pad_sents` 这个函数，来完成这个“对齐”任务吧！
    

#### **(b) (3 分) (编程题)**

> Implement the `__init__` function in `model_embeddings.py` to initialize the necessary source and target embeddings.

 这是在为我们的“大字典”注入灵魂！请在 `model_embeddings.py` 文件中，实现 `__init__` 函数。你需要在这里初始化我们的“源语言词向量矩阵”和“目标语言词向量矩阵”，它们是整个城堡的地基。
    

#### **(c) (4-分) (编程题)**

> Implement the `__init__` function in `nmt_model.py` to initialize the necessary model layers (LSTM, CNN, projection, and dropout) for the NMT system.

 现在要开始搭建城堡的“主体结构”啦！请在 `nmt_model.py` 文件中，实现 `__init__` 函数。在这里，你需要把我们 NMT 系统需要的所有“建筑模块”都准备好，包括：LSTM层、CNN层、各种投影层（线性层）以及 Dropout 层。
    

#### **(d) (8 分) (编程题)**

> Implement the `encode` function in `nmt_model.py`. This function converts the padded source sentences into the tensor X, generates henc1,…,hencmh_{\text{enc}_1}, \dots, h_{\text{enc}_m}, and computes the initial state hdec0h_{\text{dec}_0} and initial cell cdec0c_{\text{dec}_0} for the Decoder. You can run a non-comprehensive sanity check by executing: `python sanity_check.py 1d`

 这是“阅读家”（编码器）的工作时间！请在 `nmt_model.py` 中实现 `encode` 函数。这个函数负责：
    
    1. 将填充好的源语言句子转换成张量 XX。
        
    2. 生成对原文的“阅读笔记”，也就是 henc1,…,hencmh_{\text{enc}_1}, \dots, h_{\text{enc}_m}。
        
    3. 为“创作家”（解码器）计算出初始的“创作灵感” hdec0h_{\text{dec}_0} 和 cdec0c_{\text{dec}_0}。
        
- 写完后，你可以运行 `python sanity_check.py 1d`，来初步看看你的实现是不是对的。
    

#### **(e) (8 分) (编程题)**

> Implement the `decode` function in `nmt_model.py`. This function constructs yˉ\bar{y} and runs the `step` function over every timestep for the input. You can run a non-comprehensive sanity check by executing: `python sanity_check.py 1e`

 这是“创作家”（解码器）的宏观工作流程！请在 `nmt_model.py` 中实现 `decode` 函数。这个函数需要构建我们之前提到的、包含上一步信息的输入 yˉ\bar{y}，并且控制着，在每一个时间步，都去调用一次 `step` 函数来完成单步的创作。
    
- 同样，可以用 `python sanity_check.py 1e` 来进行检查！
    

#### **(f) (10 分) (编程题)**

> Implement the `step` function in `nmt_model.py`. This function applies the Decoder’s LSTM cell for a single timestep, computing the encoding of the target subword hdecth_{\text{dec}_t}, the attention scores ete_t, attention distribution αt\alpha_t, the attention output ata_t, and finally the combined output oto_t. You can run a non-comprehensive sanity check by executing: `python sanity_check.py 1f`

 这是我们整个魔法最核心、最精妙的一步！请在 `nmt_model.py` 中实现 `step` 函数。这个函数负责完成**单次**的创作步骤，包括：
    
    1. 运行解码器的LSTM单元，得到新的思考 hdecth_{\text{dec}_t}。
        
    2. 计算注意力的“相关性分数” ete_t 和“墨水浓度” αt\alpha_t。
        
    3. 得到“划好的重点摘要” ata_t。
        
    4. 最后，融合所有信息，得到“综合输出向量” oto_t。
        
- 这是最关键的一步，完成后，记得用 `python sanity_check.py 1f`检查。

