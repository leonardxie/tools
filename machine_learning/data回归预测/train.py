#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：train.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024-06-05 16:25
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import seaborn as sns
from model import Mlp, TF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义loss
loss = nn.MSELoss()

# 模型初始化
# We can experiement with the model. For example, the simplest can be a single nn.Linear model
def get_net(in_features):
    #net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64,1)).to(device)
    net = TF(in_features=in_features, drop=0.).to(device)
    return net


# ##### Log root mean squared error
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 加载数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# ##### Define trainning model
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls, net


# ##### K-fold cross-validation
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx,:], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        in_features = data[0].shape[1]
        net = get_net(in_features)
        train_ls, valid_ls, net = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
        save_model(net, f'net_{i}.pth')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    in_features = train_features.shape[1]
    net = get_net(in_features)
    train_ls, valid_ls, net = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    train_ls_all = []
    train_ls_all.append(train_ls)
    # plt.plot(np.arange(1,101,1),train_ls_all[0])
    # plt.xlabel('epoch'), plt.ylabel('rmse')
    print(f'train log rmse {float(train_ls[-1]):f}')
    preds = net(test_features).cpu().detach().numpy()
    test_data_pred_res = pd.Series(preds.reshape(1, -1)[0])
    # submission = test_data['target']
    test_data_pred_res.to_csv('submission.csv', index=False)
    save_model(net, 'net.pth')
    return test_data_pred_res


def save_model(model, model_name):
    torch.save({'model': model.state_dict()}, model_name)
    print("Save model success!")


def load_model(model_path):
    model = get_net(in_features=15)
    state_dict = torch.load('net.pth')
    model.load_state_dict(state_dict['model'])


def data_preprocess(train_raw, test_raw):
    # concat 训练和测试数据，并进行标准化
    all_features_raw = pd.concat((train_raw.iloc[:, 0:-1], test_raw.iloc[:, 0:-1]))
    all_features = all_features_raw.copy()

    # ### Data Preprocessing
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # heatmap
    # corrmat = all_features.corr()
    # plt.subplots(figsize=(15, 15))
    # sns.heatmap(corrmat, vmax=0.9, square=True)
    # plt.savefig('./corr.png')
    # plt.show()

    # ##### Get Dummies
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
    print(all_features)

    # ###### Convert to tensors
    n_train = train_raw.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_raw.target.values.reshape(-1, 1), dtype=torch.float32).to(device)

    return train_features, test_features, train_labels


def main():
    # 导入数据
    train_raw = pd.read_csv("train_data.csv")
    test_raw = pd.read_csv("test_data.csv")

    train_features, test_features, train_labels = data_preprocess(train_raw=train_raw, test_raw=test_raw)

    # 模型训练
    # k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.002, 2, 64
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.0046, 2, 512

    res = train_and_pred(train_features, test_features, train_labels, test_raw,
                   num_epochs, lr, weight_decay, batch_size)

    mae = mean_absolute_error(test_raw['target'], res)
    mse = mean_squared_error(test_raw['target'], res)
    rmse = sqrt(mse)
    r2_s = r2_score(test_raw['target'], res)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    print("test score: ", r2_s)



    # print("K-折交叉验证:\n")
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
    #                           weight_decay, batch_size)
    # print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
    #       f'avg valid log rmse: {float(valid_l):f}')

    # 学习率调整测试
    # print("\n\nTrain on different lr range from 0.01 to 0.0001 with stepping 0.0001:\n")
    # lr_range = np.arange(0.0001, 0.01, 0.0001).tolist()
    # for lr in lr_range:
    #     print('\nlr:{:.5f}'.format(lr))
    #     res = train_and_pred(train_features, test_features, train_labels, test_raw,
    #                num_epochs, lr, weight_decay, batch_size)
    #     mae = mean_absolute_error(test_raw['target'], res)
    #     mse = mean_squared_error(test_raw['target'], res)
    #     rmse = sqrt(mse)
    #     r2_s = r2_score(test_raw['target'], res)
    #     print("MAE: %.2f" % mae)
    #     print("MSE: %.2f" % mse)
    #     print("RMSE: %.2f" % rmse)
    #     print("test score: ", r2_s)


if __name__ == "__main__":
    main()