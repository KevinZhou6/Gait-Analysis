import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchvision.models.inception as inception
from torch.utils.tensorboard import SummaryWriter

def read(path,label):
    # 读取CSV文件并转换为numpy数组
    data = pd.read_csv(path, header=None)
    scaler = MinMaxScaler()
    scaled_data1 = scaler.fit_transform(data)

    data_array = np.array(scaled_data1, dtype=np.float64)
    data_label = pd.read_csv(label, header=None)
    label_array = np.array(data_label, dtype=np.float64)

    return data_array, label_array