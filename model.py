import torch
import torch.nn as nn

# 改进的 PyTorch LSTM + CNN + Transformer 模型
class CNNLSTMTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, nhead=8, num_transformer_layers=4):
        super(CNNLSTMTransformerClassifier, self).__init__()
        # 嵌入层（使用预训练嵌入）
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 多层卷积层，提取不同感受野的特征
        self.conv1d_3 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, padding=3)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 第一层 Transformer 编码器
        self.transformer_encoder_layer_1 = nn.TransformerEncoderLayer(d_model=384, nhead=nhead)
        self.transformer_encoder_1 = nn.TransformerEncoder(self.transformer_encoder_layer_1, num_layers=num_transformer_layers)
        # 多层双向 LSTM
        self.lstm1 = nn.LSTM(384, hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=0.3)
        # 第二层 Transformer 编码器
        self.transformer_encoder_layer_2 = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=nhead)
        self.transformer_encoder_2 = nn.TransformerEncoder(self.transformer_encoder_layer_2, num_layers=num_transformer_layers)
        # 添加额外的 LSTM 层
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=0.3)
        # 层级归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        # Dropout 层 (50% dropout)
        self.dropout = nn.Dropout(0.5)
        # 残差连接
        self.residual_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        # 全连接层，ReLU 激活
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        # 输出层
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8)
        # 添加自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        # 卷积操作，转置维度以适配 Conv1D
        embedded = embedded.permute(0, 2, 1)
        conv_out_3 = self.conv1d_3(embedded)
        conv_out_5 = self.conv1d_5(embedded)
        conv_out_7 = self.conv1d_7(embedded)
        # 拼接卷积输出
        conv_out = torch.cat((conv_out_3, conv_out_5, conv_out_7), dim=1)
        pooled_out = self.pool(conv_out)
        pooled_out = pooled_out.permute(2, 0, 1)  # 转换为 [seq_len, batch_size, feature_dim] 适配 Transformer
        # 第一层 Transformer 编码器
        transformer_out_1 = self.transformer_encoder_1(pooled_out)
        transformer_out_1 = transformer_out_1.permute(1, 0, 2)  # 转换回 [batch_size, seq_len, feature_dim]
        # 第一层双向 LSTM
        lstm_out, _ = self.lstm1(transformer_out_1)
        lstm_out = self.layer_norm(lstm_out)  # 添加归一化
        # 第二层 Transformer 编码器
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim * 2] 适配 Transformer
        transformer_out_2 = self.transformer_encoder_2(lstm_out)
        transformer_out_2 = transformer_out_2.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim * 2]
        # 第二层双向 LSTM
        lstm_out, _ = self.lstm2(transformer_out_2)
        # 使用多头注意力机制
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        # 残差连接
        lstm_out_residual = attn_output + self.residual_fc(lstm_out)
        # 使用最后一个时间步的隐藏状态，并应用自适应池化
        lstm_out = self.adaptive_pool(lstm_out_residual.permute(0, 2, 1)).squeeze(-1)
        lstm_out = self.dropout(lstm_out)
        # 全连接层和激活函数
        fc_out = self.fc1(lstm_out)
        fc_out = self.relu(fc_out)
        # 最终输出层
        out = self.fc2(fc_out)
        return out