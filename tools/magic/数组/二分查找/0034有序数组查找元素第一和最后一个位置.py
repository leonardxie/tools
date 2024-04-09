#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：0034有序数组查找元素第一和最后一个位置.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/23 15:26 
'''
from typing import List

class Solution1:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        :param nums: List[int]
        :param target: int
        :return: List[int]
            给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
            如果数组中不存在目标值 target，返回 [-1, -1]。

            进阶：你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
            示例 1：
            输入：nums = [5,7,7,8,8,10], target = 8
            输出：[3,4]

            示例 2：
            输入：nums = [5,7,7,8,8,10], target = 6
            输出：[-1,-1]

            示例 3：
            输入：nums = [], target = 0
            输出：[-1,-1]
        """

        """"
        寻找target在数组里的左右边界，有如下三种情况：
        情况一：target 在数组范围的右边或者左边，例如数组{3, 4, 5}，target为2或者数组{3, 4, 5},target为6，此时应该返回{-1, -1}
        情况二：target 在数组范围中，且数组中不存在target，例如数组{3,6,7},target为5，此时应该返回{-1, -1}
        情况三：target 在数组范围中，且数组中存在target，例如数组{3,6,7},target为6，此时应该返回{1, 1}
        """

        if target<nums[0] or target>nums[right]:
            return [-1, -1]

        def getLeftBorder(nums, target):
            left = 0
            right = len(nums) - 1
            left_border = -2
            while left <= right:
                middle = int((left+right)/2)
                if nums[middle] >= target:
                    right = middle - 1
                    left_border = right
                else:
                    left = middle + 1
            return left_border

        def getRightBorder(nums, target):
            left = 0
            right = len(nums) - 1
            right_border = -2
            while left <= right:
                middle = int((left+right)/2)
                if nums[middle] <= target:
                    left = middle + 1
                    left_border = left
                else:
                    right = middle - 1
            return right_border

        leftBoder = getLeftBorder(nums, target)
        rightBoder = getRightBorder(nums, target)
        # 情况一
        if leftBoder == -2 or rightBoder == -2:
            return [-1, -1]
        # 情况三
        if rightBoder - leftBoder > 1:
            return [leftBoder + 1, rightBoder - 1]
        # 情况二
        return [-1, -1]

# 解法2
# 1、首先，在 nums 数组中二分查找 target；
# 2、如果二分查找失败，则 binarySearch 返回 -1，表明 nums 中没有 target。此时，searchRange 直接返回 {-1, -1}；
# 3、如果二分查找成功，则 binarySearch 返回 nums 中值为 target 的一个下标。然后，通过左右滑动指针，来找到符合题意的区间


class Solution2:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        def binarySearch(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            while left<=right:  # 不变量：左闭右闭区间
                middle = left + (right-left) // 2
                if nums[middle] > target:
                    right = middle - 1
                elif nums[middle] < target:
                    left = middle + 1
                else:
                    return middle
            return -1
        index = binarySearch(nums, target)
        if index == -1:
            return [-1, -1]  # nums 中不存在 target，直接返回 {-1, -1}
        # nums 中存在 targe，则左右滑动指针，来找到符合题意的区间
        left, right = index, index
        # 向左滑动，找左边界
        while left - 1 >= 0 and nums[left - 1] == target:
            left -= 1
        # 向右滑动，找右边界
        while right+1 < len(nums) and nums[right + 1] == target:
            right += 1
        return [left, right]


# 解法3
# 1、首先，在 nums 数组中二分查找得到第一个大于等于 target的下标（左边界）与第一个大于target的下标（右边界）；
# 2、如果左边界<= 右边界，则返回 [左边界, 右边界]。否则返回[-1, -1]
class Solution3:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int, lower:bool) -> int:
            left, right = 0, len(nums)-1
            ans = len(nums)
            while left <= right: # 不变量：左闭右闭区间
                middle = left + (right-left) // 2
                # lower为True，执行前半部分，找到第一个大于等于 target的下标 ，否则找到第一个大于target的下标
                if nums[middle] > target or (lower and nums[middle] >= target):
                    right = middle - 1
                    ans = middle
                else:
                    left = middle + 1
            return ans

        leftBorder = binarySearch(nums, target, True)  # 搜索左边界
        rightBorder = binarySearch(nums, target, False) - 1  # 搜索右边界
        if leftBorder<= rightBorder and rightBorder< len(nums) and nums[leftBorder] == target and nums[rightBorder] == target:
            return [leftBorder, rightBorder]
        return [-1, -1]


# 解法4
# 1、首先，在 nums 数组中二分查找得到第一个大于等于 target的下标leftBorder；
# 2、在 nums 数组中二分查找得到第一个大于等于 target+1的下标， 减1则得到rightBorder；
# 3、如果开始位置在数组的右边或者不存在target，则返回[-1, -1] 。否则返回[leftBorder, rightBorder]
class Solution4:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            while left<=right: # 不变量：左闭右闭区间
                middle = left + (right-left) //2
                if nums[middle] >= target:
                    right = middle - 1
                else:
                    left = middle + 1
            return left  # 若存在target，则返回第一个等于target的值

        leftBorder = binarySearch(nums, target) # 搜索左边界
        rightBorder = binarySearch(nums, target+1) -1  # 搜索右边界
        if leftBorder == len(nums) or nums[leftBorder]!= target: # 情况一和情况二
            return [-1, -1]
        return [leftBorder, rightBorder]

