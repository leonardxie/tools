#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：worker.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 14:41 
'''
from time import ctime, sleep


class Worker:
    def __init__(self):
        pass

    def task(self, sleep_sec=10, tag='test'):
        print(f'[{tag}] start sleep {sleep_sec}, at {ctime()}')
        sleep(10 - sleep_sec)
        # print(f'[{tag}] finish sleep {sleep_sec}, at {ctime()}')

        return sleep_sec

def main():
    worker = Worker()
    print(worker.task(5))


if __name__=='__main__':
    main()