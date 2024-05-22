#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：data.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/5/11 9:39
'''

from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="./corpus"):
    """读取数据"""
    assert split in ['train', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n' and line != '\r\n':
                if '\r\n' in line:
                    if len(line) == 4:
                        word, tag = line[0], line[1]
                    else:
                        word, tag = line[0], line[2]
                else:
                    if len(line) == 3:
                        word, tag = line[0], line[1]
                    else:
                        word, tag = line[0], line[2]
                # word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def load_txt_data(filepath, make_vocab=True):
    word_lists = []
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            if line != '\n' and line != '\r\n':
                if '\r\n' in line:
                    word_list = list(line.replace('\r\n', ''))
                    word_lists.append(word_list)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        return word_lists, word2id
    else:
        return word_lists


def load_list_data(list_data, make_vocab=True):
    word_lists = []
    for single_data in list_data:
        word_list = list(single_data.replace('\r\n', ''))
        word_lists.append(word_list)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        return word_lists, word2id
    else:
        return word_lists



