#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：session.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024-04-23 9:54 
'''

from typing import List

class Session:

    def __init__(self):
        self.children = []
        self.child_is_done = []
        self.is_done = False

    def convert_node_to_session(self, node, node_list):
        self.session_id = node.index
        self.children = node.children
        self.child_is_done = [node_list[_].status for _ in node.children]
        self.is_done = node.status






