#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_threading_demo2.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:50 
'''
import threading
from time import sleep, ctime

loops = [4, 2]

class ThreadFunc(object):

    def __init__(self, func, args, name=''):
        self.name = name
        self.func = func
        self.args = args

    def __call__(self):
        self.res = self.func(*self.args)


def loop(nloop, nsec):
    print(f'start loop {nloop} at: {ctime}')
    sleep(nsec)
    print(f'loop {nloop} done at: {ctime()}')

def main():
    print(f'starting at: {ctime()}')
    threads = []
    nloops = range(len(loops))

    for i in nloops: # create all threads
        t = threading.Thread(target=ThreadFunc(loop, (i, loops[i]), loop.__name__))
        threads.append(t)

    for i in nloops: # start all threads
        threads[i].start()

    for i in nloops: # wait for all completion
        threads[i].join()

    print(f'all DONE at: {ctime()}')

if __name__ == '__main__':
    main()



