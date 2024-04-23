#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：node.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024-04-23 10:29 
'''

"""
构建基本单位node
"""

from typing import List
from pydantic import BaseModel

class Node(BaseModel):
    index: int
    content: int
    children: List[int]
    status: bool
