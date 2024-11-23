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
        * 新能力
            * 冷启动推荐：TIGER能够有效推荐新加入的数据集中的item，提升冷启动场景下的推荐效果
            * 推荐多样性：通过调整生成过程中的温度参数，TIGER能够控制推荐结果的多样性，增加用户体验的丰富性
5. 结论
    * 创新点：提出了一种结合`生成式检索`和`语义ID表示`的推荐框架TIGER，实现更好的规模和泛化能力
    * 实验验证：在多个真实世界的数据集上验证了TIGER的有效性，展示了其在`推荐准确性`、`冷启动`和`多样性`方面的优势
6. 总结
    * 通过引入生成式检索和语义ID表示，突破了传统推荐系统的局限，提升了推荐的准确性和泛化能力。TIGER框架不仅在标准推荐任务中表现优异，还在冷启动和推荐多样性等实际应用场景中展现出显著优势，为推荐系统的发展提供了新的思路和方法。



## Adapting LLMs by Integrating Collaborative Semantics for Recommendation
通过集成`推荐的协作语义`来调整LLMs
* <a href="./papers/Adapting-LLMs-by-Integrating-Collaborative-Semantics-for-Recommendation.pdf">查看PDF</a>
* <a href="https://github.com/rucaibox/lc-rec">codes</a>