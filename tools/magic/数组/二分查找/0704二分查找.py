#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：0704二分查找.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/18 14:14 
'''

class Solution(object):
    def search(self, nums, target):
        """
        :param nums: List[int]
        :param target: int
        :return: int
            给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target，
            写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
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
        return -1  # 未找到目标值








