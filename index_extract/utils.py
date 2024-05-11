#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：utils.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/5/11 9:39
'''
import pickle
import joblib


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def save_model_joblib(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        joblib.dump(model, f)


def load_model_joblib(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = joblib.load(f)
    return model
