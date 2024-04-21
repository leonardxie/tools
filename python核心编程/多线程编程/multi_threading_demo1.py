#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_threading_demo1.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:43 
'''

import threading
from time import sleep, ctime

loops = [4, 2]

def loop(nloop, nsec):
    print(f'start loop {nloop}, at: {ctime()}')
    sleep(nsec)
    print(f'loop {nloop} done at: {ctime()}')

def main():
    print(f'starting at:{ctime()}')
    threads = []
    nloops = range(len(loops))

    for i in nloops:
        t = threading.Thread(target=loop, args=(i,loops[i]))
        threads.append(t)

    for i in nloops: # start threads
        threads[i].start()

    for i in nloops: # wait for all threads to finish
        threads[i].join()

    print(f"all DONE at: {ctime()}")


if __name__ == "__main__":
    main()