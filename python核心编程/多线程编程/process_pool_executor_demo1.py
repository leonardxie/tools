#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：process_pool_executor_demo1.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 9:39 
'''
from time import ctime, sleep
from concurrent.futures import ProcessPoolExecutor, as_completed

def task(sleep_sec=10, tag='test'):
    print(f'[{tag}] start sleep {sleep_sec}')
    sleep(10-sleep_sec)
    print(f'[{tag}] finish sleep {sleep_sec}')
    return sleep_sec


def main():
    # executor = ProcessPoolExecutor(max_workers=5)
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10):
            single_process = executor.submit(task, i, tag='TEST')
            futures.append(single_process)

        for future in as_completed(futures):
            data = future.result()
            print('result is %s' % str(data))

    # executor.shutdown()


if __name__ == '__main__':
    main()

