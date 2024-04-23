#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：thread_pool_executor_facfib.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/21 20:44 
'''


import time

import threading
import concurrent
import concurrent.futures
from time import sleep, ctime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures._base import Future

def compute(num):
    print(f'执行函数 function {num}，即将sleep {num}秒')
    sleep(num)
    return f'函数{num}'




def mian():
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i in range(10):
            futures.append(executor.submit(compute, i))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                print(result)


if __name__ == '__main__':
    mian()