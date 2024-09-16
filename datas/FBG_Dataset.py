import torch
from torch.utils.data import Dataset, DataLoader


# 创建自定义数据集
class FBGDataset(Dataset):
    def __init__(self, x_tensor, y_position_tensor, y_force_tensor):
        self.x_tensor = x_tensor
        self.y_position_tensor = y_position_tensor
        self.y_force_tensor = y_force_tensor

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        x = self.x_tensor[idx]
        y_position = self.y_position_tensor[idx]
        y_force = self.y_force_tensor[idx]
        return x, y_position, y_force


if __name__ == '__main__':

    # 创建数据集
    x_tensor = torch.randn(504, 2000)
    y_position_tensor = torch.randn(501, 1)
    y_force_tensor = torch.randn(504, 1)
    dataset = FBGDataset(x_tensor, y_position_tensor, y_force_tensor)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 查看 DataLoader 的输出
    for batch_idx, (inputs, labels_position, labels_force) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Inputs:", inputs.shape)
        print("Labels (Position):", labels_position.shape)
        print("Labels (Force):", labels_force.shape)
        break  # 示例中只打印第一个批次
