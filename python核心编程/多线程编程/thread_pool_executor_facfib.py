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
import queue
import threading
import functools
import traceback
import concurrent
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import Future

def  compute_a(index):
    print(f'a， index：{index}')
    return 'compute_a'

def compute_b(index):
    print(f'b， index：{index}')
    return 'compute_b'

def compute_c(index):
    print(f'c， index：{index}')
    return 'compute_c'


functions = [compute_a, compute_b, compute_c]
futures = []

with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    for i in range(10):
        futures.append(executor.submit(functions, i))

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            print(result)






