#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：evaluate.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/5/11 9:39
'''

from utils import load_model, load_model_joblib


def crf_evaluate(model_path, test_word_lists):
    # moel_path = "./ckpts/crf.joblib"
    crf_model = load_model_joblib(model_path)

    # 验证CRF模型
    pred_tag_lists = crf_model.test(test_word_lists)

    test_data = ["".join(_) for _ in test_word_lists]
    res = ["".join(_) for _ in pred_tag_lists]
    return_res = []
    for i in range(len(res)):
        single_pred = res[i]
        length = len(single_pred.replace('O', ''))
        if length > 0:
            # print(f'query：{test_data[i]}\n抽取结果：{test_data[i][:length]}\n\n')
            return_res.append(test_data[i][:length])
        else:
            # print(f'query：{test_data[i]}\n抽取结果：无\n\n')
            return_res.append("")

    return return_res


