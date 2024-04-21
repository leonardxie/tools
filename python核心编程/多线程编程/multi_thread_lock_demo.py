#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_thread_lock_demo.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:15 
'''

import _thread
from time import sleep, ctime

loops = [4, 2]


def loop(nloop, nsec, lock):
    print(f'start loop {nloop}, at: {ctime()}')
    sleep(nsec)
    print(f'loop {nloop} done at: {ctime()}')
    lock.release()


def main():
    print(f'starting at:{ctime()}')
    locks = []
    nloops = range(len(loops))
    for i in nloops:
        lock = _thread.allocate_lock() # 获得线程本地锁
        lock.acquire() # 开始加锁，获得锁并加锁
        locks.append(lock) # 向当前的锁集合中添加该锁

    for i in nloops:
        _thread.start_new_thread(loop, (i, loops[i], locks[i]))# 启动2个线程来执行loop函数并传递参数

    # 反复检查锁是否被所著，如果被锁住就一直死循环，否则停止循环检查
    for i in nloops:
        # 最后会阻塞当前的线程，反复检查当前的锁是否被锁住，如果被锁住就暂停等待解锁，才能让主线程停止
        while locks[i].locked():pass

    # 当所有的线程都执行完毕后就会执行最后的打印
    print(f"all DONE at:{ctime()}")


if __name__ == '__main__':
    main()
