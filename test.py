import torch

def test_one_epoch(model, dataloader, criterion_position, criterion_force, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    total_batches = len(dataloader)

    # 禁用梯度计算
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels_direction, labels_position, labels_force = data
            inputs, labels_direction, labels_position, labels_force = inputs.to(device), labels_direction.to(device), labels_position.to(device), labels_force.to(device)
        
             # 前向传播
            outputs_direction, outputs_position, outputs_force = model(inputs)
            # 计算损失
            loss_direction = criterion_position(outputs_direction, labels_direction)
            loss_position = criterion_position(outputs_position, labels_position)
            loss_force = criterion_force(outputs_force.squeeze(), labels_force)
            loss = loss_position + loss_force + loss_direction

            # 累计损失
            running_loss += loss.item()

    # 计算整个 epoch 的平均损失
    epoch_loss = running_loss / total_batches
    print(f'Test Epoch Loss: {epoch_loss:.10f}')
    return epoch_loss
