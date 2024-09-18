import torch
from datas.FBG_Dataset import min_max_denormalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(data_loader, model, model_path):
        for batch_idx, (inputs, labels_position, labels_force) in enumerate(data_loader):
            print(f"Batch {batch_idx + 1}:")
            print("Inputs:", inputs.shape)
            # 加载模型参数
            model.load_state_dict(torch.load(model_path))
            # 将模型设置为评估模式
            model.eval()
            # 进行预测
            with torch.no_grad():
                position_output, force_output = model(inputs.to(device))
            print("Real_Labels (Position):", labels_position)
            print("Predicted position:", torch.argmax(position_output, dim=1))
            print("Real_Labels (Force):", min_max_denormalize(labels_force, 0, 20))
            print("Predicted force:", min_max_denormalize(force_output, 0, 20))
            break  # 示例中只打印第一个批次