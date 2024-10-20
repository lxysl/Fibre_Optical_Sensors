import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 创建自定义数据集
class FBGDataset(Dataset):
    def __init__(self, x_tensor, y_direction_tensor ,y_position_tensor, y_force_tensor):
        self.x_tensor = x_tensor
        self.y_direction_tensor = y_direction_tensor
        self.y_position_tensor = y_position_tensor
        self.y_force_tensor = y_force_tensor

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        x = self.x_tensor[idx]
        y_direction = self.y_direction_tensor[idx]
        y_position = self.y_position_tensor[idx]
        y_force = self.y_force_tensor[idx]
        return x, y_direction, y_position, y_force
    
def z_score_normalize_samplewise(data):
    means = np.mean(data, axis=1,keepdims = True)  # 每个样本的均值
    print(means.shape)
    stds = np.std(data, axis=1,keepdims = True)    # 每个样本的标准差
    print(stds.shape)
    return (data - means) / stds

# Min-Max归一化函数
def min_max_normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def min_max_denormalize(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val


if __name__ == '__main__':

    # 创建数据集
    x_tensor = torch.randn(504, 2000, 2)
    y_position_tensor = torch.randn(501, 1)
    y_force_tensor = torch.randn(504, 1)
    y_direction_tensor = torch.randn(504, 1)
    dataset = FBGDataset(x_tensor,  y_direction_tensor, y_position_tensor, y_force_tensor)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 查看 DataLoader 的输出
    for batch_idx, (inputs, label_direction ,labels_position, labels_force) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Inputs:", inputs.shape)
        print("Labels (Position):", labels_position.shape)
        print("Labels (Force):", labels_force.shape)
        print("Labels (Direction):", label_direction.shape)
        break  # 示例中只打印第一个批次
