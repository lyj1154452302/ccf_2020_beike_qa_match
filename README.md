# 比赛

贝壳找房-房行业聊天问答匹配，比赛地址：https://www.datafountain.cn/competitions/474

# 写在前面

这是我第一次正式参加一个大型的NLP比赛，全程solo，经过一个多月的努力，最终结果是A榜70+，B榜60+。虽然结果不够理想，但在整个过程中学习到了很多知识，踩了不少坑，在赛后也认识到了很多不足的地方。因此决定在这个仓库内总结回顾一下自己的一些思路方法，代码也相应的共享出来。第一次在github上记录，有些地方写的像流水账，轻喷QAQ。如果有幸能帮到你，那还请顺手点个star吧~

本项目都是基于pytorch实现的。

# 数据

首先来说说数据，此次赛题是以房产中介为背景，客户与中介的问答文本作为数据，客户的一个问题对应n个候选回答，这其中有0~n个候选回答是这个问题的正确回答。

# 数据处理

由于BERT里的token是字符级别的，因此我没有做分词，停用词和标点符号我也没有做处理，因为我觉得部分停用词和标点符号在问答场景下可以提供特定的信息。最终，我只是将question和reply做成了pair对。

![数据处理](img\数据处理.png)

# 预训练模型

BERT for ever~本项目用的都是BERT，后期模型融合的时候用到了RoBERTa。

BERT：[bert-base-chinese](https://huggingface.co/bert-base-chinese)

RoBERTa: [chinese-roberta-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)

# baseline

这些baseline都是在我对这个赛题和这类问题的认知不断进化的过程中而得出来的，因此，baseline也由简到繁。

### 1. 最简单的NSP fine-tune方案

看到这个赛题的时候，我第一反应就是，这不就是bert中的nsp任务吗，于是我翻看了huggingface的文档，找到了他们包装好的最简单的Bert对于NSP任务的fine-tune方法：[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)，仔细翻看源码后会发现，其实就是将input喂入预训练好的BERT后，得到pooled output（即BERT NSP任务分类器的隐含层输出）后，再接一层自定义的新的dense层来分类。此方案经过5折取平均后，测试集的效果在0.7495

### 2. BERT后接pooling

这个baseline是看了包大人的分享后学到的。其思路就是将input喂入预训练好的BERT后，得到sequence_output（每个token在BERT最后一层的hidden state）后，对其进行averaging pooling和max pooling，这么做的原因，我认为是把整个序列经过bert计算后，把所有词的语义向量压缩成一个句向量，用句向量来表达整句话的语义。在得到两个pooling向量后，包大人还把最后一层的[CLS]token的hidden state和最后一个token（亦即第二句话末尾的[SEP]）的hidden state与两个pooling向量拼接，最终再通过一层dense进行分类。此方案经过5折取平均后，测试集的效果在0.7606。

### ---------------------------

看了夕小瑶大神的文章里提到的**Siamese**结构后，链接在这：https://zhuanlan.zhihu.com/p/87384188。开始尝试**Siamese**，我尝试多个结构来作为sentence encoder，分别有：BERT的最后一层输出、BILSTM、CNN。下面几个baseline就是围绕这些来的。

<img src="img\Simaese.png" alt="Simaese" style="zoom:50%;" />

### 3. BERT输出层取句向量做交互

**Siamese**结构，句向量由bert**最后一层**或**倒数第二层**输出来取句向量，**这里句向量是以max pooling**的方式得到的，然后做交互，具体的交互方式主要是**两个句向量相乘、相减**。图1方案经过5折取平均后，测试集的效果在0.758。图2方案经过5折取平均后，测试集的效果在0.77，可见不同层级的语义表示可以提供更多的语义信息。其中，seq_1表示question，seq_2表示reply。

<img src="img\BERTSimaese.png" alt="BERTSimaese" style="zoom: 50%;" />

<center>图1</center>

<img src="D:\PyCharmProjects\ccf_2020_beike_qa_match\img\BERTSimaese-2.png" alt="BERTSimaese-2" style="zoom: 50%;" />

<center>图2</center>

### 4. BERT后接BILSTM（整个序列encode），再取句向量做交互

**Siamese**结构，bert后接BILSTM，注意，这里是对**整个序列**（question和reply组成的pair）做encode，之后根据BILSTM的输出取句向量，**这里取句向量的方式是max pooling**，然后做交互，具体的交互方式主要是两个句向量相乘、相减。图1方案经过5折取平均后，测试集的效果在0.768。图2方案经过5折取平均后，测试集的效果在0.773，可见将BERT最后一层的输出与LSTM得到的输出一起用于句向量，有助于提升效果。其中，seq_1表示question，seq_2表示reply。

<img src="img\Simaese BERT+LSTM.png" alt="Simaese BERT+LSTM" style="zoom: 50%;" />

<center>图1</center>

<img src="D:\PyCharmProjects\ccf_2020_beike_qa_match\img\Simaese BERT+LSTM-2.png" alt="Simaese BERT+LSTM-2" style="zoom:50%;" />

<center>图2</center>

### 5. BERT后接BILSTM（question和reply分别encode），再取句向量做交互

**Siamese**结构，bert后接BILSTM，注意，这里是对question和reply**分别encode**，这里的BILSTM是**tied weigt**的（encode question用的BILSTM和encode reply用的BILSTM是同一个，即权重绑定的）。之后根据BILSTM的输出取句向量，注意，这里取句向量的方式变了，是**取BILSTM中双向的最后一个时间步的hidden state**，也就是$[h^{->}_{n}; h^{<-}_{0}]$，然后做交互，具体的交互方式主要是两个句向量相乘、相减。其中，seq_1表示question，seq_2表示reply。该方案经过5折取平均后，测试集的效果在0.772。

这个结构来自于这篇文章 [Siamese Recurrent Architectures for Learning Sentence Similarity](https://www.researchgate.net/publication/307558687)。这个文章里也解释了什么情况下要用tied weight，什么情况下不用：当sentence1和sentence2的文本是非对称域（如搜索中的query和document的域就不对称）的时候使用两个不同的sentence encoder，效果会更好。在这个比赛里，question和reply的文本属于同一领域，所以使用tied weight版。

<img src="img\Simaese BERT+LSTM+TIEDWEIGHT.png" alt="Simaese BERT+LSTM+TIEDWEIGHT" style="zoom:60%;" />

### 6. BERT后接CNN（question和reply分别encode），再取句向量做交互

**Siamese**结构，bert后接CNN（准确来说是TextCNN），注意，这里是对question和reply**分别encode**，这里的CNN是**tied weigt**的（encode question用的CNN和encode reply用的CNN是同一个，即权重绑定的）。之后根据CNN的输出取句向量，这里取句向量的方式是max pooling，然后做交互，具体的交互方式主要是两个句向量相乘、相减。其中，seq_1表示question，seq_2表示reply。该方案经过5折取平均后，测试集的效果在0.772。

<img src="D:\PyCharmProjects\ccf_2020_beike_qa_match\img\Simaese BERT+CNN+TIEDWEIGHT.png" alt="Simaese BERT+CNN+TIEDWEIGHT" style="zoom:50%;" />

### 7. BERT后接TextCNN

bert后接TextCNN，该方案经过5折取平均后，测试集的效果在0.772。

<img src="img\BERTTextCNN.png" alt="BERTTextCNN" style="zoom:50%;" />

# 大模型

在baseline都调到最佳参数后，我才上了large模型，效果提升在0.5 ~1%。

# 模型融合

我只尝试了下面两个融合策略：

1. **stacking**，融合了各种方案各种预训练模型的组合，效果提升在0.5%左右。
2. **voting**，融合了各种方案各种预训练模型的组合，效果提升在1%左右，效果比stacking好。

# 数据增强

本次比赛尝试过的方法：

1. 回译

用的是[百度API](https://api.fanyi.baidu.com/doc/21)，我选择了**中->英->中**、**中->法>中**，只增加中->英->中到原始数据集里的话，效果提升在0.5%~1%，但是同时把中->英->中和中->法>中都加入到原始数据集的话，效果反而下降了，猜测原因：两次回译结果的语义差不多，因此都加到原始数据集里的话，语义相同的句子过多，从而导致模型过拟合。

2. EDA（Easy Data Augmentation）

对于一个样本，采用**同义词替换**、**随机插入**、**随机交换**和**随机删除**三种方法进行增强，用的是github上的开源工具包 [EDA_NLP_for_Chinese](https://github.com/zhanlaoban/EDA_NLP_for_Chinese)，蛮好用的，源码也好理解，可以根据需要自己修改代码。效果的话...一言难尽，和回译放在一起，效果反而下降了。

# 用过的一些trick

1. BN（适合在CNN、Dense里用）

2. LN（适合在RNN里用）

3. 权重初始化

   1）kaiming初始化，适合用于激活函数为ReLU函数

   2）Xavier初始化，适合用于激活函数为tanh函数

4. word embedding层后加dropout（据说可以缓解过拟合）

5. 最佳f1阈值搜索，即在**验证集**的所有预测概率结果下，找到使得**验证集F1指标**最高的一个阈值**threshold**，之后在测试集预测概率结果下，若正样本的预测概率大于threshold，则预测为1，否则为0。这种trick找到了全局最优的F1值，[参考链接](https://www.jianshu.com/p/51debab91824)。

6. **post pretrain**，即利用赛题的语料，利用NSP任务以及MLM任务来进一步的训练预训练模型。我用的是 [UER](https://github.com/dbiir/UER-py) ，它是腾讯开源的一个工具包，可以预训练模型、fine-tune各种下游任务，非常方便。

# 踩过的坑

虽然有些坑都很基础（说明自己是真的菜0.0），但是都是真实碰到过的，也都挨个解决了，这个过程很有意义，收获很多，因此记录一下，大佬勿嘲。

有些坑记录的有些啰嗦，各位按需跳过~

1. validation代码中忘记添加 `with torch.no_grad()` 导致validation的时候**显存爆炸**。

2. 加载模型权重时，注意组件的名字要**对齐**，否则就会出现保存文件里的权重名称和代码里的模型类的模型组件对不上，导致读取失败。

3. 有一天突然发现，相同的模型权重，相同的测试集，**预测的结果却不一样**。后面发现是自己使用dropout时有问题，我在模型中用的是`x = nn.functional.dropout(x, p=0.5)`，因为这个函数的一个参数**training**默认是**True**，在预测阶段的时候这个参数没有被改变，才会出现每次预测都不一样的情况。若要继续用这个函数，则需要根据模型的train模式和和eval模式指定training参数的状态，个人觉得有些麻烦且容易遗漏，所以**更推荐的做法**是在模型初始化的时候建立一个**dropout层**，即`nn.Dropout(p=0.5)`，它封装了`nn.functional.dropout`这个函数，且**训练状态**也会根据model的状态来改变。问题顺利解决，最后附上解决这个问题的[参考链接](https://blog.csdn.net/junbaba_/article/details/105673998)。

4. 早停策略，最初设计的时候，是以valid_loss为判断标准，也就是连续n个epoch，valid_loss都没有比当前最低值下降，则停止训练。但我后面发现，虽然valid_loss没有下降，但是valid_auc值还在不断上升，而我们保存的缺失valid_loss最低值时的模型，这时的模型的valid_auc比后面几个epoch的valid_auc要低，于是陷入了一个问题：该选择valid_loss最低时候的模型（情况A）还是该选择valid_auc最低时候的模型（情况B）呢？随后我从实际情况出发，去看了一下这两个情况下保存的模型在测试集上的表现，结果是情况B的模型在测试集上的表现比情况A的模型要好的多，因此我后面选择了情况B作为早停策略的设计。

   之后呢，我又仔细的思考了一个问题：按理说，valid_loss开始不断上升，与train_loss的差距不断拉大，说明模型的过拟合程度在逐渐加大，但是valid_auc却仍在上升，说明其在验证集上的实际表现越来越好，感觉前后有些自相矛盾。后面我看到了知乎上的这个 [问题](https://www.zhihu.com/question/318399418) 的高赞回答，才知道出现这个情况的原因是，模型对一些预测错误的样本过于极端（自信）导致的，使得这些样本的loss很高，从而使得这些预测错误的样本的loss主导（dominate）了整个loss。

   我也问过一个大佬，他说实际情况下，一般都是以验证集的表现（各种评价指标）来衡量模型的能力，而不是用loss。

# 遗憾

有一些一直想尝试但因为一些原因而没有尝试的东西

1. **focal loss**，一方面可以缓解样本不平衡对模型训练的影响，另一方面可以让模型更加注重那些难样本。
2. **LazyAdam**（pytorch中对应的方法叫SparseAdam），由于现在大部分都是各种fine-tune BERT，在fine-tune的过程中也会更新词向量矩阵，而在 [这篇博客](https://www.jianshu.com/p/48e71b72ca67) 中提到一个问题，由于NLP的稀疏性（即大部分词或字出现的频率很低，少部分词或字出现的频率较高），在使用Adam的时候会导致那些原本梯度应该为0的embedding，由于动量的存在，而有了非0梯度，从而加重了过拟合的程度。
3. **Bad case分析**。这方面没什么经验，所以当我打印出那些模型预测错误的样本时，分析不出什么规律0.0。后来逛知乎时，看到一个答主说bad case分析，说到 “如果没有什么规律，但是发现模型高置信度做错的这些样本大部分都是标注错误的话，就直接把这些样本都删掉，常常可以换来性能的小幅提升，毕竟测试集都是人工标注的，困难样本和错标样本不会太多。“，顿时觉得好有道理，而且仔细观察这个赛题的数据你会发现，确实有一些样本的标注是明显有问题的。

# 不足的地方

1. 要说不足的地方，给我冲击最大的就是代码能力、组织架构能力太拉胯了。在初期，由于代码写的不够鲁棒，导致我后面扩展的时候一直在写重复的代码，做重复的事情。要知道，少做重复的事情，才能大大提升效率。
2. 不够细心，有好几次因为自己的疏忽，导致浪费了很多时间，比如train了一天的模型，才发现代码有问题，真是太气啦。这也让我明白了一个道理，在开始大规模（时间成本、空间成本较大的情况）的运作之前，一定要先用小规模的内容做个测试，小规模都跑通了没问题了，只是扩大内容规模的话一般就不会有问题了。

3. 实验记录做的不够好。虽然这次从初期开始就有用MindMaster（一个思维导图的软件）去做好记录，但是由于是第一次，经验还是不足，有些实验和一些想法记录的很粗糙，导致后面回看的时候有点懵逼想不起来。另外，实验记录真的很重要，当你没有思路的时候，重新回看实验记录，能够带给你很大的灵感。特别是在调参的时候，如果没有实验记录，非常容易混乱掉。
4. 最后也是最重要的一点，回过头来想，我对数据的了解少之又少，没有花什么时间在了解数据上面，可能这就是包大人谈到的 [数据敏感性](https://zhuanlan.zhihu.com/p/335363661) 吧，相信如果对数据有充分的了解，对后续方案的改进会提供不小的帮助，后续要加强这方面。

# 一些好的东西，后续要延续下去

1. 实验记录，用思维导图记录会清楚一些
2. 调参的时候，尽量把所有参数都写在一个参数列表里，这样调起来才快，而不是到不同的代码块去找到参数后再修改，这样既乱又容易遗漏。
3. 遇到问题及时做记录，及时总结。