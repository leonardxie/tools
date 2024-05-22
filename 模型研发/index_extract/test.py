#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：test.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/5/11 9:39
'''

from data import build_corpus, load_txt_data, load_list_data
from evaluate import crf_evaluate

CRF_MODEL_PATH = 'ckpts/crf.joblib'


def main():
    print("读取数据...")
    # 测试脚本一：test.char.bmes
    # test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    # 测试脚本二：test.txt
    test_word_lists = load_txt_data("./corpus/test.txt", make_vocab=False)
    # 测试脚本三：["1.1. 用户登录", "1.2. 首次登陆修改密码配置"]
    # data = ["1.1. 用户登录", "1.2. 首次登陆修改密码配置", "200.5元，全部上交", "200.1234亿元"]
    # test_word_lists = load_list_data(data, make_vocab=False)
    # 测试脚本四："1.1. 用户登录"
    # data = "1.2. 首次登陆修改密码配置"
    # if isinstance(data, str):
    #     test_word_lists = load_list_data([data], make_vocab=False)

    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    res = crf_evaluate(model_path="ckpts/crf.joblib", test_word_lists=test_word_lists)
    print(res)


if __name__ == "__main__":
    main()
