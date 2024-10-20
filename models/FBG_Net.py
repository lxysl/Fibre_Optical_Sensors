import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 模型架构
# 定义一个卷积神经网络，全连接神经网络，和任务学习，输出力的位置（分类任务）和大小（回归任务）
# 分类任务利用交叉熵损失函数，回归任务利用均方误差损失函数

class FBGNet(nn.Module):
    def __init__(self):
        super(FBGNet, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 497, 128)  # 64个通道，每个通道497个特征点
        # 输出层
        self.fc_position = nn.Linear(128, 24)  # 用于分类任务：24个位置
        self.fc_force = nn.Linear(128, 1)  # 用于回归任务：力的大小

    def forward(self, x):
        # 输入x的形状为 (batch_size, 2000)，需要reshape为 (batch_size, 1, 2000)
        x = x.unsqueeze(1)
        # 卷积层 + ReLU + 池化层
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 998)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 497)
        # 展平
        x = x.view(-1, 64 * 497)  # (batch_size, 64*497)
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))  # (batch_size, 128)

        # 输出位置分类和力大小回归
        position_output = self.fc_position(x)  # (batch_size, 24)

        force_output = self.fc_force(x)  # (batch_size, 1)
        return position_output, force_output

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, num_classes_1= 25, num_classes_2= 24, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 497, 128)  # 64个通道，每个通道497个特征点
        # 输出层

        self.classification_head_1 = nn.Sequential(
        nn.Dropout(p=dropout),  # Add dropout before classification head
        nn.Linear(497, num_classes_1)
    )

        self.classification_head_2 = nn.Sequential(
        nn.Dropout(p=dropout),  # Add dropout before classification head
        nn.Linear(497, num_classes_2)
    )

        self.regression_head = nn.Sequential(
        nn.Dropout(p=dropout),  # Add dropout before regression head
        nn.Linear(497, 1)
    )
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_proj(src)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)
        output = output.permute(0, 2, 1) # (batch_size, d_model, seq_len)
        output = self.pool(F.relu(self.conv1(output)))  # (batch_size, 128, 998)
        output = self.pool(F.relu(self.conv2(output)))  # (batch_size, 256, 497)
        # Global average pooling
        output = output.mean(dim = 1)  # (batch_size, 497)
        
        # Classification task
        direction_output = self.classification_head_1(output)
        position_output = self.classification_head_2(output)
        
        # Regression task
        reg_output = self.regression_head(output).squeeze(-1)
        
        return direction_output, position_output, reg_output




if __name__ == '__main__':
    

    input_dim = 2
    num_classes = 5  # 假设有5个类别

    model = MultiTaskTransformer(input_dim = input_dim)
    x = torch.randn(32, 2000, 2)
    direction_output, position_output, reg_pred = model(x)

    print(f"Classification output shape: {direction_output.shape}")
    print(f"Regression output shape: {position_output.shape}")
    print(f"Regression output shape: {reg_pred.shape}")

    # print(model)
    from torchinfo import summary
    summary(model, input_size=(32, 2000, 2))  # do a test pass through of an example input size