from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from utils import time_calc


def train_one_epoch(model, dataloader, criterion_position, criterion_force, optimizer, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    total_batches = len(dataloader)

    # 遍历数据集的每一个批次
    for i, data in enumerate(dataloader, 0):
        inputs, labels_position, labels_force = data
        inputs, labels_position, labels_force = inputs.to(device), labels_position.to(device), labels_force.to(device)
        # 重置梯度
        optimizer.zero_grad()
        # 前向传播
        outputs_position, outputs_force = model(inputs)
        # 计算损失
        loss_position = criterion_position(outputs_position, labels_position)
        loss_force = criterion_force(outputs_force.squeeze(), labels_force)
        loss = loss_position + loss_force
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 计算总损失
        running_loss += loss.item()

    # 输出整个 epoch 的平均损失
    epoch_loss = running_loss / total_batches
    tqdm.write(f'Epoch Loss: {epoch_loss:.10f}')
    return epoch_loss

@time_calc
def train(model, dataloader, num_epochs, criterion_position, criterion_force, optimizer, device):
    for epoch in range(num_epochs):
        # 调用 train_one_epoch 进行训练
        epoch_loss = train_one_epoch(model, dataloader, criterion_position, criterion_force, optimizer, device)
        tqdm.write(f'Epoch {epoch + 1} finished with loss: {epoch_loss:.3f}')

    print('Finished Training')