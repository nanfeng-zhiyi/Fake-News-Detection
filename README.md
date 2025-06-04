# 虚假新闻检测 (Fake-News-Detection)

本项目旨在通过深度学习模型检测中文虚假新闻。

![](E:\git项目\Fake-News-Detection\images\dection.png)

*   **`model.py`**: 定义了核心的深度学习模型 `CNNLSTMTransformerClassifier`。该模型是一个混合架构，具体包含：
    *   **词嵌入层 (Embedding Layer)**: 将文本词汇转换为向量表示。
    *   **多尺度一维卷积层 (Multi-Scale 1D CNN)**: 使用不同大小的卷积核 (3, 5, 7) 提取局部文本特征。
    *   **最大池化层 (Max Pooling)**: 对卷积输出进行降维。
    *   **两层Transformer编码器 (Dual Transformer Encoders)**: 用于捕捉文本中的长距离依赖关系。
    *   **两层双向长短期记忆网络 (Dual Bidirectional LSTMs)**: 进一步处理序列信息，捕捉上下文依赖。
    *   **多头注意力机制 (Multi-Head Attention)**: 增强模型对关键信息的关注。
    *   **残差连接 (Residual Connections)**、**层归一化 (Layer Normalization)** 和 **Dropout**: 用于提升模型性能和泛化能力。
    *   **全连接输出层**: 进行最终的分类判断。
*   **`run_gpu.py`**: 该脚本负责整个流程的执行，包括：
    *   **数据加载与预处理**:
        *   从 CSV 文件 (`combined.csv`, `train.csv`) 加载数据。
        *   使用 `BeautifulSoup` 清洗HTML内容，并进行文本正则化。
        *   使用 `jieba` 进行中文分词。
        *   构建词典，将文本转换为数字序列，并进行填充 (padding) 处理。
    *   **模型训练 (`train` 模式)**:
        *   在 GPU 环境下训练 `CNNLSTMTransformerClassifier` 模型。
        *   使用 `AdamW` 优化器和 `CrossEntropyLoss` 损失函数。
        *   集成 `ReduceLROnPlateau` 学习率调度策略。
        *   通过 `TensorBoard` 记录训练过程中的损失和准确率。
        *   根据验证集上的 F1 分数保存表现最佳的模型。
    *   **模型预测 (`predict` 模式)**: 使用训练好的模型对新的文本进行预测。
    *   **模型评估**: 在验证集上计算 F1 分数和准确率。
*   **`combined.csv`**: 主要的训练数据集。

# 结果

最终总榜排名在人工智能赛道取得了第26名的成绩，f1分数为0.985左右

![](E:\git项目\Fake-News-Detection\images\result.png)



