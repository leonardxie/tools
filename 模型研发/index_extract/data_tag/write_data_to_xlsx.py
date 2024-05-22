#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：write_data_to_csv.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/5/11 14:09 
将待打标签的数据写入xlsx文件
'''

from openpyxl import Workbook, load_workbook
import json
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

def main():
    data_positive = []
    data_negative = []
    with open('./corpus/data_add.txt', "r", encoding="utf-8") as f:
        for i in f.readlines():
            data_positive.append(list(i.strip()))

    # 创建一个新的工作簿
    wb = Workbook()

    # 选择默认的活动工作表
    ws = wb.active


    # 遍历
    for row in data_negative:
        # centered_cell = [cell.set_style(alignment) for cell in row]
        ws.append(row)

        ws.append([])

    # # 保存工作簿为excel文件
    pos_file_name = '../data_tag/data.xlsx'
    wb.save(pos_file_name)

    # 关闭工作簿
    wb.close()


if __name__ =='__main__':
    main()