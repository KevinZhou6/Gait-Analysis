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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, dropout=0.5):
        super(CNN_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_dim
        self.conv_layer = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.lstm_layer = nn.LSTM(64, hidden_dim, num_layers ,dropout=0.3,batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, output_size)
        self.attention_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # CNN层
        x = x.float()
        # print("输入时：")
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print("第一次变换后：")
        # print(x.shape)
        x = self.conv_layer(x)
        # print("卷积输出：")
        # 调整形状
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print("第二次调整后：")
        # print(x.shape)
        # LSTM层
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm_layer(x, (h0, c0))

        # Attention层
        attention_scores = self.attention_layer(out)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        attention_output = attention_weights * out
        context_vector = torch.sum(attention_output, dim=1)
        # 全连接层
        output = self.fc_layer(context_vector)
        # out = self.fc_layer(out[:, -1, :])
        output1 = nn.functional.sigmoid(output)
        return output1
