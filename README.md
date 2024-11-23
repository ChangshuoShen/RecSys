# Recommender Systems


## Recommender Systems with Generative Retrieval
使用生成检索的推荐系统
* <a href="./papers/Recommender-Systems-with-Generative-Retrieval.pdf">查看PDF</a>
* <a href="https://github.com/EdoardoBotta/RQ-VAE-Recommender">codes</a>

### 主要内容
1. 引言
* 推荐系统重要性
* 现有方法的局限
    * 传统的**检索-排序**策略依赖于高维嵌入和ANN搜索，面对大规模项目时，存储和计算成本较高，且难以适应新项目的快速增长。
* 此工作贡献
    * TIGER (Transformer Index for GEnerative Recommenders) 框架，通过生成检索`直接预测项目的语义ID`，利用Transformer的记忆力作为End2End的索引，来克服传统方法的局限
2. 相关工作
    * 序列推荐模型：GTU4REC、NARM、SASRec、BERT4Rec
    * 语义ID生成：
        * 讨论VQ-Rec以及其他生成语义ID的方法
        * 强调本文使用RQ-VAE (Residual-Quantized Variational AutoEncoder)
    * 生成式检索
        * 回顾GENRE、DSI等生成式检索方法
        * 指出本文首次将生成式检索应用于RecSys，并结合语义ID表示item
3. 方法
    * 框架：TIGER分为两阶段
        1. 语义ID生成：使用预训练内容编码器（如Sentence-T5）将项目的`内容特征`转化为`语义嵌入`，然后通过RQ-VAE两化生成语义ID
        2. 生成式推荐模型训练：训练一个基于Transformer的seq2seq模型，输入用户的`历史语义ID`信息，预测下一个item的语义ID
    * 语义ID生成细节
        * RQ-VAE：采用`残差量化方法`，将语义嵌入`分层量化`为多个代码词，形成语义ID，确保相似的item具有相似的语义ID
        * 处理冲突：通过`在语义ID末尾添加额外的标记`，确保每个item具有唯一的语义ID
    * 生成式检索
        * 将用户的item交互历史转换为语义ID序列，使用Transformer模型预测下一个item的语义ID
        * 通过seq2seq的生成过程，实现End2End的的推荐，不依赖于预先构建的嵌入序列
4. 实验
    * 数据集：选用了Amazon Product Reviews dataset中的“Beauty”、“Sports and Outdoors”和“Toys and Games”三个类别，包含`用户评论`和`项目元数据`。
    * 评价指标：
        * Recall@K
            * 在推荐列表的前K个item中，实际感兴趣的item被正确推荐的比例
            * $Recall@K = \frac{\#推荐列表中世界感兴趣项目}{\#用户实际感兴趣项目的总数量}$
            * 反映RecSys在捕捉用户兴趣方面的`覆盖能力`
        * NDGG(Normalized Discounted Cumulative Gain)@K
            * 归一化折扣累计增益，衡量推荐列表中推荐项目的相关性及其排序质量的指标
            * $NDCG@K = \frac{DCG@K}{IDCG@K}$
                * $DCG@K = \Sigma_{i = 1}^K \frac{rel_i}{log_2(i + 1)}$
                * $IDCG@K = \Sigma_{i = 1}^K \frac{rel_i^{ideal}}{log_2(i + 1)}$
            * 表示推荐列表中相关性高的item是否被排在前面
        * (K = 5、10)
    * 结果
        * 性能对比：TIGER在所有三个数据集上的表现均显著优于当时的Baseline方法，如GRU4Rec、SASRec、BERT4Rec等，尤其在NDCG@5和Recall@5上提升幅度最大。
        * 语义ID的有效性：通过Ablation Study验证RQ-VAE生成的语义ID相比于随机ID和LSH(Locality Sensitive Hashing)生成的ID具更好的性能
        * 新能力一、
* <a href="./papers/Adapting-LLMs-by-Integrating-Collaborative-Semantics-for-Recommendation.pdf">查看PDF</a>
* <a href="https://github.com/rucaibox/lc-rec">codes</a>

### 主要内容
0. 摘要
    * LLMs在推荐系统中展现出巨大潜力，能够`提升现有推荐模型`或`作为骨干结构`使用
    * 存在LLMs与推荐系统之间的语义鸿沟，因推荐的项目通常使用LLM词汇表之外的离散标识符（如item ID）
    * 研究问题：如何有效整合`语言语义`和`协同语义`，以充分利用LLMs的模型能力用于推荐任务。
    * 提出方法: LC-Rec(Language and Collaborative semantics for improving LLMs in Recommender systems)
        * 项目索引：设计了一种基于学习的向量量化方法，结合统一的语义映射，为项目分配有意义且不冲突的ID（项目索引）
        * 对齐微调：提出一系列专门设计的微调任务（微调任务包括`顺序项目预测`、`显式索引-语言对齐`和`隐式推荐导向对齐`），以增强LLMs中协同语义的整合
    * 贡献：
        * LC-Rec模型能够在生成推荐时`直接从整个项目集合中生成项目`，无需依赖候选集
        * 设计基于`向量量化`和`统一语义映射`的项目索引方法，确保项目索引的语义意义和唯一性
        * 提出了多种`语义对齐微调任务`，深度整合语言和协同语义，提升LLMs在推荐系统中的适应性

1. 引言
    * 推荐系统根据用户偏好`动态演变`，顺序推荐通过`捕捉用户行为的序列特征`受到广泛关注
    * 传统推荐模型多基于用户互动日志的item ID序列，忽略了LLMs的语言语义能力
    * LLMs与推荐系统之间的语义鸿沟限制了其在推荐任务中的应用效果
2. 相关工作
    * 顺序推荐: 介绍了基于RNN、CNN、GNN、Transformer等深度神经网络的顺序推荐方法，以及利用项目内容信息（如标题、描述）的增强方法。
    * LLMs在推荐中的应用: 讨论了将LLMs适配于推荐系统的两种主要方法：\
        * 文本序列表示用户行为
        * 基于项目ID的生成推荐，但存在各自的局限性。

3. 方法
    * 整体方法概述:
        * 项目索引: 采用`向量量化方法`基于LLMs编码的项目文本嵌入学习离散索引，并通过`统一语义映射`减轻索引分配冲突。
        * 对齐微调: 设计多种微调任务，包括`顺序项目预测`、`显式索引-语言对齐`和`隐式推荐导向对齐`，深化语言与协同语义的整合。
    * 学习项目索引:
        * 使用Residual-Quantized Variational AutoEncoder (RQ-VAE)进行多层向量量化，生成项目索引
        * 引入统一语义映射通过`Sinkhorn-Knopp算法`，确保最后索引层的语义均匀分布，避免索引冲突
    * 语言与协同语义的对齐:
        * 顺序项目预测: 将推荐任务转化为生成式任务，基于用户历史交互索引序列预测下一项目索引。
        * 显式索引-语言对齐: 通过指令让LLM根据项目标题/描述生成对应的索引，反之亦然，增强索引与语言语义的关联。
        * 隐式推荐导向对齐:
            * 异步项目预测：改变条件与目标的表示形式，提高语义对齐难度。
            * 基于用户意图的项目预测：从用户评价中提取意图，生成符合用户需求的项目索引。
            * 个性化偏好推断：基于索引序列推断用户显式偏好。
    * 训练与推理:
        * 使用LLaMA 7B作为基础模型，通过指令微调优化模型。
        * 生成推荐时，采用束搜索在索引令牌间进行，确保生成的索引合法。
    * 讨论:
        * 将LC-Rec与现有基于语言模型的推荐方法进行对比，强调其在语义整合和索引机制上的优势。

4. 实验
    * 实验设置:
        * 数据集: 使用Amazon三个子集（乐器、艺术与工艺、视频游戏），数据稀疏性高。
        * 基线模型: 包括Caser, HGN, GRU4Rec, BERT4Rec, SASRec, FMLP-Rec, FDSA, S3-Rec, P5-CID, TIGER等。
        * 评估指标: 使用HR@K和NDCG@K（K=1,5,10），采用留一法进行全排名评估。
        * 实现细节: 项目索引构建、模型优化参数、微调策略等。
    * 整体性能:
        * LC-Rec在三个数据集上均表现最佳，与基线方法相比有显著提升，平均性能提升达25.5%。
    * 消融研究:
        * 不同语义对齐任务的影响: 逐步加入不同对齐任务（SEQ, MUT, ASY, ITE, PER）显著提升性能。
        * 不同项目索引方法的比较: LC-Rec的索引方法优于传统的单一ID、随机索引以及去除统一语义映射的版本。
    * 进一步分析:
        * 基于用户意图的项目预测: LC-Rec在理解项目索引语义上优于DSSM等基线方法。
        * 嵌入可视化分析: PCA展示了LC-Rec将项目索引有效整合进LLM的语义空间。
        * 语义相似负样本上的性能: LC-Rec在区分语义相似的负样本上表现优异，证明了语言与协同语义的有效整合。
    * 案例研究:
        * 分析多层索引中语义信息的逐步细化过程，以及基于索引生成相关项目的效果，展示了LC-Rec在语义整合上的优势。
5. 结论
    * 提出了LC-Rec，一种基于LLMs的推荐方法，重点在项目索引和对齐微调两个方面。
    * 通过向量量化与统一语义映射学习项目索引，并设计多种语义对齐任务，成功将协同语义整合进LLMs。
    * 实验结果证明LC-Rec在顺序推荐任务中优于多种竞争基线方法。
    * 未来工作包括在多轮对话设置中扩展方法，以及在领域适应时保留LLMs的通用能力。

6. 主要贡献总结
    * 模型创新: 提出LC-Rec，通过向量量化结合统一语义映射学习项目索引，确保了索引的语义一致性和唯一性。
    * 语义整合: 设计多种微调任务，深度整合语言语义与协同语义，提升了LLMs在推荐系统中的适应性和性能。
    * 实验验证: 在多个真实数据集上验证了LC-Rec的有效性，显著超越传统和现有的LLM-based推荐方法。
    * 应用潜力: LC-Rec不仅适用于顺序推荐，还可扩展至其他推荐任务，如捆绑预测和解释生成，具有广泛的应用前景。
7. 研究意义
    * 有效弥合大型语言模型与推荐系统之间的语义鸿沟，充分发挥了LLMs在推荐任务中的潜力。
    * 提供了一种通用的语义整合框架，可为未来基于LLMs的推荐系统研究提供参考和基础。