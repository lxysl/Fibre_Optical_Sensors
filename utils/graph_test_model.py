import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import FBGNet , MultiTaskTransformer
from datas.FBG_Dataset import min_max_denormalize
from datas import FBGDataset, z_score_normalize_samplewise, min_max_normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data = np.loadtxt('/home/chengjun/Fibre_Optical_Sensors/Data_Sets/test_data.txt', delimiter=',')  # (19968, 2000)
print(x_data.shape)
# 步骤1：重塑数组
x_data = x_data.reshape(60, 2, 2000)
# 步骤2：调整轴的顺序
normalized_data_x = np.transpose(x_data, (0, 2, 1))

y = np.loadtxt('/home/chengjun/Fibre_Optical_Sensors/Data_Sets/text_label.txt', delimiter=',')  # (9984, 3)

# 假设 x 和 y 是 numpy 数组，需要转换为 PyTorch 的张量
x_tensor = torch.from_numpy(normalized_data_x).float()  # 输入数据
y_direction_tensor = torch.from_numpy(y[:, 0]).long()
y_position_tensor = torch.from_numpy(y[:, 1]).long()
y_force_tensor = torch.from_numpy(y[:,2]).float()

# 创建数据集实例
fbg_dataset = FBGDataset(x_tensor, y_direction_tensor, y_position_tensor, y_force_tensor)
test_dataloader = DataLoader(fbg_dataset,batch_size=60,)
model = MultiTaskTransformer(input_dim = 2)
model = nn.DataParallel(model)
model.to('cuda')

def test_model(data_loader, model, model_path):
        for batch_idx, (inputs, label_direction, labels_position, labels_force) in enumerate(data_loader):
            print(f"Batch {batch_idx + 1}:")
            print("Inputs:", inputs.shape)
            # 加载模型参数
            model.load_state_dict(torch.load(model_path))
            # 将模型设置为评估模式
            model.eval()
            # 进行预测
            with torch.no_grad():
                direction_output,position_output, force_output = model(inputs.to(device))
            # print("Real_Labels(Direction)", label_direction)
            # print("Predicted position:", torch.argmax(direction_output, dim=1))
            print("Real_Labels(Direction)-Predicted position",label_direction.to('cuda')-torch.argmax(direction_output, dim=1))
            # print("Real_Labels (Position):", labels_position - 1)
            # print("Predicted position:", torch.argmax(position_output, dim=1))
            print("Real_Labels (Position)-Predicted position",(labels_position - 1).to('cuda')-torch.argmax(position_output, dim=1))
            # print("Real_Labels (Force):", labels_force)
            # print("Predicted force:", force_output)
            print('Real_Labels - Predicted force:' ,labels_force.to('cuda') - force_output)
            break  # 示例中只打印第一个批次

if __name__ == '__main__':
    test_model(test_dataloader,model,'/home/chengjun/Fibre_Optical_Sensors/optical_fiber_checkpoints/model_transformer_regularization_2000.pth')