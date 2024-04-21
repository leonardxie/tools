#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_thread_demo.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:05 
'''
import _thread
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
    print(f'starting at:{ctime()}')
    _thread.start_new_thread(loop0, ())
    _thread.start_new_thread(loop1, ())
    sleep(6)
    print(f"all DONE at:{ctime()}")


if __name__ == '__main__':
    main()
