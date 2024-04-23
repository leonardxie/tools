#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：tread_pool_executor_demo6.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 11:10 
'''

from time import ctime, sleep
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
# from multiprocessing import Condition

trigger_nums = {
    4: [1, 2, 3],
    7: [5, 6],
    9: [8]
}
finish_nums = []
def need_to_wait(sleep_sec, trigger_nums, finish_nums, condition):
    res = True
    if sleep_sec in list(trigger_nums.keys()):
        for condition_num in list(trigger_nums[sleep_sec]):
            if condition_num not in finish_nums:
                res = False
    return res

def task(sleep_sec=10, condition=None, trigger_nums={}, finish_nums = [], tag='test'):
    with condition:
        condition.wait_for(lambda: need_to_wait(sleep_sec, trigger_nums, finish_nums, condition))
        print(f'[{tag}] start sleep {sleep_sec}')
        sleep(sleep_sec)
        print(f'[{tag}] finish sleep {sleep_sec}')
        finish_nums.append(sleep_sec)
        condition.notify_all()
    return 100


def main():
    confition = threading.Condition()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10):
            single_process = executor.submit(task, i, confition, trigger_nums, finish_nums, tag='TEST')
            futures.append(single_process)

    for future in as_completed(futures):
        data = future.result()
        print('result is %s' % str(data))

    executor.shutdown()


if __name__ == '__main__':
    main()
