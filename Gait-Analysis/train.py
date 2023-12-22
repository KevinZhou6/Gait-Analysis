import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchvision.models.inception as inception
from torch.utils.tensorboard import SummaryWriter
from data import *
from network import CNN_LSTM

# 读入
train_data_array,train_label_array = read('train_data.csv','train_label.csv')
test_data_array,test_label_array = read('test_data.csv','test_label.csv')
val_data_array,val_label_array = read('val_data.csv','val_label.csv')

# 将numpy数组转换为3维形状
train_data_3d = np.reshape(train_data_array, (100, 40, 30))
train_dataset_tensor = torch.from_numpy(train_data_3d)
train_label_tensor = torch.from_numpy(train_label_array)

test_data_3d = np.reshape(test_data_array, (32, 40, 30))
test_dataset_tensor = torch.from_numpy(test_data_3d)
test_label_tensor = torch.from_numpy(test_label_array)

val_data_3d = np.reshape(val_data_array, (32, 40, 30))
val_dataset_tensor = torch.from_numpy(val_data_3d)
val_label_tensor = torch.from_numpy(val_label_array)

# 将numpy数组转换为3维形状
train_data_3d = np.reshape(train_data_array, (100, 40, 30))
train_dataset_tensor = torch.from_numpy(train_data_3d)
train_label_tensor = torch.from_numpy(train_label_array)

test_data_3d = np.reshape(test_data_array, (32, 40, 30))
test_dataset_tensor = torch.from_numpy(test_data_3d)
test_label_tensor = torch.from_numpy(test_label_array)

val_data_3d = np.reshape(val_data_array, (32, 40, 30))
val_dataset_tensor = torch.from_numpy(val_data_3d)
val_label_tensor = torch.from_numpy(val_label_array)
print(train_data_3d.shape)
print(train_label_array.shape)
print(test_dataset_tensor.shape)
print(test_label_tensor.shape)
print(val_dataset_tensor.shape)
print(train_label_tensor.shape)
# print(train_label_tensor)

# 通过DataLoader按批处理数据
# 对数据进行封装：(数据，标签)
train_dataset = TensorDataset(train_dataset_tensor, train_label_tensor)
val_dataset = TensorDataset(val_dataset_tensor, val_label_tensor)
test_dataset = TensorDataset(test_dataset_tensor, test_label_tensor)

batch_size = 32
# 批处理
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# data1, label1 = next(iter(test_loader))
# print(data1.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化超参数
input_size = 30# CNN输入通道数
output_size = 1 # 输出
hidden_dim = 128# 隐藏层节点个数
num_layers = 2 # lstm的层数

model1 = CNN_LSTM(input_size, hidden_dim, output_size, num_layers)

print(model1)
criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
optimizer = torch.optim.Adam(model1.parameters(), lr=0.00017)  # 优化器
num_epochs = 300# 循环次数
model = model1.to(device)

# 定义训练模型
def train(model, device, data_loader, criterion, optimizer, num_epochs, val_loader):
    historyval = []
    historytrain = []
    historyvalacc = []
    historytrainacc = []
    for epoch in range(num_epochs):
        train_loss = []
        train_correct = 0.0
        model.train()
        for data, target in data_loader:
            data = data.to(device)  # 部署到device
            target = target.to(device)
            optimizer.zero_grad()  # 梯度置零
            output = model(data)  # 模型训练
            loss = criterion(output, target.float())  # 计算损失
            train_loss.append(loss.item())  # 累计损失
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_correct += torch.sum(torch.round(output) == target)  # 比较
            trainacc = train_correct / len(data_loader.dataset)
            historytrainacc.append(trainacc.item())
        # 模型验证
        model.eval()
        val_loss = []
        val_correct = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                preds = model(data)  # 验证
                loss = criterion(preds, target.float())  # 计算损失
                val_loss.append(loss.item())  # 累计损失
                # print('pred:')
                # print(preds)
                # print('target:')
                # print(target)
                pred01 = torch.round(preds)
                val_correct += torch.sum(pred01 == target)  # 比较
                valacc = val_correct / len(val_loader.dataset)
                historyvalacc.append(valacc.item())
                print("acc")
                print(valacc)
                print("loss")
                print(val_loss)
                # print(preds)
                # print(target)
                historyval.append(np.mean(val_loss))
                # history['val_correct'].append(np.mean(val_correct))
                historytrain.append(np.mean(train_loss))
                # history['train_correct'].append(np.mean(train_correct))
        print(
            f'Epoch {epoch}/{num_epochs} --- train loss {np.round(np.mean(train_loss), 5)} --- val loss {np.round(np.mean(val_loss), 5)} --- val correct {val_correct}  --- train correct {train_correct}')

    plot1 = plt.plot(historyval)
    plot2 = plt.plot(historytrain)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend(["Training Loss", "Validation Loss"], loc=1)
    plt.show()

    plot3 = plt.plot(historyvalacc)
    plot4 = plt.plot(historytrainacc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend(["Training Acc", "Validation Acc"], loc=2)
    plt.show()


train(model, device, train_loader, criterion, optimizer, num_epochs, val_loader)

# 测试
def test(model, data_loader, device, criterion):
    test_losses = []
    num_correct = 0
    # 初始化隐藏状态
    model.eval()
    for i, dataset in enumerate(data_loader):
        data = dataset[0].to(device)  # 部署到device
        target = dataset[1].to(device)
        output= model(data)  # 测试
        loss = criterion(output, target.float())  # 计算损失
        pred = torch.round(output)  # 将预测值进行四舍五入，转换为0 或 1
        test_losses.append(loss.item())  # 保存损失
        correct_tensor = pred.eq(target)
        correct = correct_tensor.cpu().numpy()
        result = np.sum(correct)
        num_correct += result
        print("target : ", target)
        print("pred : ", pred)
        print("num correct : ", num_correct)
        print(f'Batch {i}')
        print(f'loss : {np.round(np.mean(loss.item()), 3)}')
        print(f'accuracy : {np.round(result / len(data), 3) * 100} %')
        print()
    print("测试平均损失 test loss : {:.2f}".format(np.mean(test_losses)))
    print("测试准确率 test accuracy : {:.2f}".format(np.mean(num_correct / len(data_loader.dataset))))


test(model, test_loader, device, criterion)

