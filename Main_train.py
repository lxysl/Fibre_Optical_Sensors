import os
import argparse # 用于解析命令行参数
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datas import FBGDataset, z_score_normalize_samplewise, min_max_normalize
from models import FBGNet , MultiTaskTransformer
from config import MODEL_SAVE_DIR, NUM_EPOCHS
from train import train_one_epoch
from test import test_one_epoch
from utils import test_model

# start a new wandb run to track this script
wandb.init(
    project="Fibre_Optical_sensors",
    notes="归一化了数据和力,600次迭代,加LSTM",
    config={
        "num_epochs": NUM_EPOCHS,
        "checkpoint_path": MODEL_SAVE_DIR,
        "learning_rate": 0.001,
        "architecture": "分类＋回归",
        "dataset": "output_noise_new.xlsx"
    }
)

def config_params():
    # 定义一个解析器
    parser = argparse.ArgumentParser(description="Training script to set model save path.")
    # checkpoint路径
    parser.add_argument('--checkpoint_path', type=str, default=MODEL_SAVE_DIR,
                        help="The path to the checkpoint file.")
    # 添加一个参数用于接收模型路径
    parser.add_argument('--model_save_path', type=str, default="model_both_normalize_600.pth",
                        help="The path to save the model file.")
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help="The path to save the model file.")

    args = parser.parse_args()
    return args



x_data = np.loadtxt('data.txt', delimiter=',')  # (19968, 2000)
# 步骤1：重塑数组
x_data = x_data.reshape(9984, 2, 2000)
# 步骤2：调整轴的顺序
normalized_data_x = np.transpose(x_data, (0, 2, 1))

y = np.loadtxt('label.txt', delimiter=',')  # (9984, 3)

# 假设 x 和 y 是 numpy 数组，需要转换为 PyTorch 的张量
x_tensor = torch.from_numpy(normalized_data_x).float()  # 输入数据
y_direction_tensor = torch.from_numpy(y[:, 0]).long()
y_position_tensor = torch.from_numpy(y[:, 1]).long()
y_force_tensor = torch.from_numpy(y[:,2]).float()


def main():
    arg = config_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建数据集实例
    fbg_dataset = FBGDataset(x_tensor, y_direction_tensor, y_position_tensor, y_force_tensor)
    # 设定划分比例，例如 80% 训练集，20% 测试集
    train_size = int(0.8 * len(fbg_dataset))
    test_size = len(fbg_dataset) - train_size
    # 使用 random_split 进行数据集划分
    train_dataset, test_dataset = random_split(fbg_dataset, [train_size, test_size])
    # 使用 DataLoader 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 实例化模型
    model = MultiTaskTransformer(input_dim = 2, num_classes = 21).to(device)
    # 损失函数和优化器
    criterion_position = nn.CrossEntropyLoss()  # 位置的分类损失
    criterion_force = nn.MSELoss()  # 力的大小的回归损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(arg.num_epochs):
        print(f"Epoch {epoch + 1}")
        # 调用 train_one_epoch 进行训练
        train_results = train_one_epoch(model, train_dataloader, criterion_position, criterion_force, optimizer, device)
        # print(f'Epoch {epoch + 1} finished with loss: {train_results:.3f}')
        test_results = test_one_epoch(model, test_dataloader, criterion_position, criterion_force, device)
        # print(f'Epoch {epoch + 1} finished with loss: {test_results:.3f}')
        # log the results to wandb
        wandb.log({
            "train_loss": train_results,
            "test_loss": test_results
        })

    # 保存模型
    model_path = os.path.join(arg.checkpoint_path, arg.model_save_path)
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    print("Model saved to model.pth")
    # # 测试模型
    test_model(test_dataloader, model, model_path)

    wandb.finish()



if __name__ == "__main__":
    main()