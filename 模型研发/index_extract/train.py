#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/5/11 11:11 
'''
from data import build_corpus
from models.crf import CRFModel
from utils import save_model, save_model_joblib


def crf_train(train_word_lists, train_tag_lists):

    # 训练CRF模型
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    # save_model(crf_model, "./ckpts/crf.pkl")
    save_model_joblib(crf_model, "./ckpts/crf.joblib")


def main():
    """训练模型"""
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")

    # 训练评估CRF模型
    print("训练CRF模型...")
    crf_train(train_word_lists, train_tag_lists)


if __name__ == "__main__":
    main()