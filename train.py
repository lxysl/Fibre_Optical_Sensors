from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from utils import time_calc
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


def train_one_epoch(model, dataloader, criterion_position, criterion_force, optimizer, device, mixup_idx_sample_rate):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    running_accuracy_direction = 0.0
    running_accuracy_position = 0.0
    running_mse_force = 0.0
    running_mae_force = 0.0
    total_batches = len(dataloader)

    # 遍历数据集的每一个批次
    for i, data in enumerate(dataloader, 0):
        idx, inputs, labels_direction, labels_position, labels_force = data
        inputs, labels_direction, labels_position, labels_force = inputs.to(device), labels_direction.to(device), labels_position.to(device), labels_force.to(device)
        # c-mixup
        idx_2 = np.array([np.random.choice(np.arange(len(dataloader.dataset)), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in idx])
        data_2 = dataloader.dataset[idx_2]
        idx_2, inputs_2, labels_direction_2, labels_position_2, labels_force_2 = data_2
        inputs_2, labels_direction_2, labels_position_2, labels_force_2 = inputs_2.to(device), labels_direction_2.to(device), \
                                                                          labels_position_2.to(device), labels_force_2.to(device)
        # softlabel: c -> 1-hot
        soft_labels_direction = F.one_hot(labels_direction, num_classes=25)
        soft_labels_position = F.one_hot(labels_position, num_classes=24)
        soft_labels_direction_2 = F.one_hot(labels_direction_2, num_classes=25)
        soft_labels_position_2 = F.one_hot(labels_position_2, num_classes=24)
        lambd = np.random.beta(1, 1)
        mixed_inputs = lambd * inputs + (1 - lambd) * inputs_2
        mixed_labels_direction = lambd * soft_labels_direction + (1 - lambd) * soft_labels_direction_2
        mixed_labels_position = lambd * soft_labels_position + (1 - lambd) * soft_labels_position_2
        mixed_labels_force = lambd * labels_force + (1 - lambd) * labels_force_2
        # 重置梯度
        optimizer.zero_grad()
        # 前向传播
        outputs_direction, outputs_position, outputs_force = model(mixed_inputs)
        # 计算损失
        loss_direction = criterion_position(outputs_direction, mixed_labels_direction)
        loss_position = criterion_position(outputs_position, mixed_labels_position)
        loss_force = criterion_force(outputs_force.squeeze(), mixed_labels_force)
        loss = 1.0 * loss_direction + 1.0 * loss_position + 1.0 * loss_force
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 计算总损失
        running_loss += loss.item()

        # 计算准确率 和 mse
        predicted_direction = torch.max(outputs_direction, 1)[1].cpu().numpy()
        predicted_position = torch.max(outputs_position, 1)[1].cpu().numpy()
        outputs_force = outputs_force.squeeze().detach().cpu().numpy()
        labels_direction = labels_direction.cpu().numpy()
        labels_position = labels_position.cpu().numpy()
        labels_force = labels_force.cpu().numpy()
        accuracy_direction = accuracy_score(labels_direction, predicted_direction)
        accuracy_position = accuracy_score(labels_position, predicted_position)
        mse_force = mean_squared_error(outputs_force, labels_force)
        mae_force = mean_absolute_error(outputs_force, labels_force)

        running_accuracy_direction += accuracy_direction
        running_accuracy_position += accuracy_position
        running_mse_force += mse_force
        running_mae_force += mae_force

    # 输出整个 epoch 的平均损失
    epoch_loss = running_loss / total_batches
    epoch_accuracy_direction = running_accuracy_direction / total_batches
    epoch_accuracy_position = running_accuracy_position / total_batches
    epoch_mse_force = running_mse_force / total_batches
    epoch_mae_force = running_mae_force / total_batches
    tqdm.write(f'Epoch Loss: {epoch_loss:.10f}, Accuracy Direction: {epoch_accuracy_direction:.4f}, Accuracy Position: {epoch_accuracy_position:.4f}, MSE Force: {epoch_mse_force:.4f}, MAE Force: {epoch_mae_force:.4f}')
    return epoch_loss, epoch_accuracy_direction, epoch_accuracy_position, epoch_mse_force, epoch_mae_force

@time_calc
def train(model, dataloader, num_epochs, criterion_position, criterion_force, optimizer, device):
    for epoch in range(num_epochs):
        # 调用 train_one_epoch 进行训练
        epoch_loss, epoch_accuracy_direction, epoch_accuracy_position, epoch_mse_force, epoch_mae_force = train_one_epoch(model, dataloader, criterion_position, criterion_force, optimizer, device)
        tqdm.write(f'Epoch {epoch + 1} finished with loss: {epoch_loss:.3f}, Accuracy Direction: {epoch_accuracy_direction:.4f}, Accuracy Position: {epoch_accuracy_position:.4f}, MSE Force: {epoch_mse_force:.4f}, MAE Force: {epoch_mae_force:.4f}')

    print('Finished Training')