#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：single_thread_demo.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:02 
'''
from time import sleep, ctime
def loop0():
    print(f'start loop 0, at: {ctime()}')
    sleep(4)
    print(f'loop 0 done at: {ctime()}')

def loop1():
    print(f'start loop 1, at: {ctime()}')
    sleep(2)
    print(f'loop 1 done at: {ctime()}')

def main():
    print(f'starting at: {ctime()}')
    loop0()
    loop1()
    print(f'all DONE at: {ctime()}')


if __name__ == '__main__':
    main()