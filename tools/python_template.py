#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：python_template.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/13 14:33 
'''

import sys
import os

debug = True


class FooClass(object):
    """Foo class"""
    pass


def test():
    """test function"""
    foo = FooClass()
    if debug:
        print('ran test()')


if __name__=='__main__':
    test()
