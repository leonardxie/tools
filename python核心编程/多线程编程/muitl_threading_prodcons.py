#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：muitl_threading_prodcons.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 16:07 
'''

from random import randint
from time import sleep
from queue import Queue
from myThread import MyThread

def writeQ(queue):
    print(f"producing object for Q...")
    queue.put('xxx', 1)
    print(f"size now {queue.qsize()}")

def readQ(queue):
    val = queue.get(1)
    print(f"consumed object from Q... size now, {queue.qsize()}")

def writer(queue, loops):
    for i in range(loops):
        writeQ(queue)
        sleep(randint(1, 3))

def reader(queue, loops):
    for i in range(loops):
        readQ(queue)
        sleep(randint(2, 5))

funcs = [writer, reader]
nfuncs = range(len(funcs))

def main():
    nloops = randint(2, 5)
    q = Queue(32)

    threads = []
    for i in nfuncs:
        t = MyThread(funcs[i], (q, nloops), funcs[i].__name__)
        threads.append(t)

    for i in nfuncs:
        threads[i].start()

    for i in nfuncs:
        threads[i].join()

    print("all DONE")

if __name__ == '__main__':
    main()

