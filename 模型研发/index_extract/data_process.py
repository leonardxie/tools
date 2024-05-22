#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：data_process.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/5/11 14:25 
'''

# 数据预处理，如将txt文件软化为训练格式的数据
# 格式要求 txt文件{"text": ["3", "、", "用", "户", "操", "作", "手", "册"], "label": ["B", "O", "O", "O", "O", "O", "O", "O"]}
# convert_data_to_train_bmes(file_path="./corpus/test.txt", save_path="./corpus/train.char.bmes")

def load_data(path):
    """
    读取txt文件, 加载训练数据
    :param path:
    :return:
    [
        {
        'text': ['当', '希', '望', ...],
        'label': ['O', 'O', 'O', ...]
        },
        {'text': ['当', '希', '望', ...],
        'label': ['O', 'O', 'O', ...]
        }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]


def add_data_to_train_bmes(file_path, save_path):
    data = load_data(file_path)

    with open(save_path, 'a', encoding='utf-8') as f:
        for single_data in data:
            word_list = single_data['text']
            tag_list = single_data['label']

            for i, char in enumerate(word_list):
                line = word_list[i] + ' ' + tag_list[i]
                f.write(line + '\n')
            f.write('\n')
        f.write('\n')


def create_data_to_bmes(file_path, save_path):
    data = load_data(file_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        for single_data in data:
            word_list = single_data['text']
            tag_list = single_data['label']

            for i, char in enumerate(word_list):
                line = word_list[i] + ' ' + tag_list[i]
                f.write(line + '\n')
            f.write('\n')
        f.write('\n')


def main():
    add_data_to_train_bmes(file_path="./corpus/train.txt", save_path="corpus/train.char.bmes")
    # create_data_to_bmes(file_path="./corpus/train.txt", save_path="./corpus/train.char.bmes")


if __name__ == "__main__":
    main()