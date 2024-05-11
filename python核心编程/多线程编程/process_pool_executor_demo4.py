#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：process_pool_executor_demo4.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024-04-25 11:41 
'''


from time import ctime, sleep
from concurrent.futures import ProcessPoolExecutor, as_completed

def task(index, sleep_sec=5):
    # print(f'[{tag}] start sleep {sleep_sec}')
    sleep(sleep_sec)
    print(f'at {ctime()}, index {index}, finish sleep {sleep_sec}\n')
    return sleep_sec

def play(executor, index):
    futures = []
    for i in range(10):
        single_process = executor.submit(task, index)
        # futures.append(single_process)

    # for future in as_completed(futures):
    #     data = future.result()
    #     print(f'index {index}, result is {data}')

def main():
    # executor = ProcessPoolExecutor(max_workers=5)
    executor = ProcessPoolExecutor(max_workers=15)
    for index in range(10):
        play(executor, index)
    # executor.shutdown()


if __name__ == '__main__':
    main()
