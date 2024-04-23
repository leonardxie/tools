#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：process_pool_executor_demo3.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 14:56 
'''

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：process_pool_executor_demo2.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/4/22 9:58
'''

import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import multiprocessing

trigger_nums = {
    4: [1, 2, 3],
    7: [5, 6],
    9: [8]
}

finish_nums = []


class A:

    def __init__(self):
        self.executor = None

    def task(self, num):
        time.sleep(num)
        return num

    def step_one(self, num_list):
        i = 1
        result_list = []
        for result in  self.executor.map(self.task, num_list):
            print("task{}:{}".format(i, result))
            i += 1
            result_list.append(result)
        return result_list

    def step_two(self, num_list):
        i = 1
        result_list = []
        for result in self.executor.map(self.task, num_list):
            print("task{}:{}".format(i, result))
            i += 1
            result_list.append(result)

        return result_list

    def start(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        res = self.step_one([1, 2, 4, 2])
        print(res)
        res = self.step_two([1, 2, 4, 2])
        print(res)
        self.executor.shutdown()



if __name__ == '__main__':
    start = time.time()
    A().start()
    end_time = time.time()
    print(end_time-start)
