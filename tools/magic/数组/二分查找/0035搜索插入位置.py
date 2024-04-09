#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：0035搜索插入位置.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/19 9:42 
'''
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
        :param nums: List[int]
        :param target: int
        :return: int
            给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
            你可以假设数组中无重复元素。

            示例 1:
            输入: [1,3,5,6], 5
            输出: 2

            示例 2:
            输入: [1,3,5,6], 2
            输出: 1

            示例 3:
            输入: [1,3,5,6], 7
            输出: 4

            示例 4:
            输入: [1,3,5,6], 0
            输出: 0
        """
        left = 0
        right = len(nums)-1 # 定义target在左闭右闭的区间里，[left, right]
        while left <= right:
            # middle = int((left+right)/2)  # 向下取整
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle - 1  # target在左区间，所以[left, middle - 1]
            elif nums[middle] < target:
                left = middle + 1  # target在右区间，所以[middle + 1, right]
            else:
                return middle  # 数组中找到目标值，直接返回下标
        return right + 1  # 未找到目标值