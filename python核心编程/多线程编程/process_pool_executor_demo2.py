#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：process_pool_executor_demo2.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 9:58 
'''

from time import ctime, sleep, time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import multiprocessing
import queue
from worker import Worker


class Controller:
    def __init__(self):
        self.queue_waiting: queue.Queue[int] = queue.Queue()
        self.queue_executing: queue.Queue[int] = queue.Queue()
        self.queue_finish: queue.Queue[int] = queue.Queue()

        self.worker = Worker
        self.execute_func = self.worker.task

        self.check_rule = {
            4: [1, 2, 3],
            7: [5, 6],
            9: [8]
        }
        self.executor = ProcessPoolExecutor(max_workers=5)

        self.worker_thread = threading.Thread(target=self.cache_consumer, daemon=True)
        self.worker_thread.start()

    def cache_consumer(self):
        sleep(1)
        print(f'对象初始化')
        return 0

    def compute_a(self, index):
        sleep(index)
        return f'用户 {index} finish a function'

    def compute_b(self, index):
        # print( f"用户 {index} 调用大模型")
        # 现有方案：串行调用大模型
        # for i in range(1, 11):
        #     self.compute_llm(index, i) # 10个用户，并发运行10次，每个用户调10次大模型，总耗时 126s
        # 优化方案2：多线程调用大模型 # 10个用户，并发运行10次，每个用户并发调10次大模型，总耗时 46s
        futures_llm = []
        with ThreadPoolExecutor(max_workers=5) as executor_tmp:
            print('start llm')
            for i in range(1, 11):
                futures_llm.append(executor_tmp.submit(self.compute_llm, index, i))
            for futures_llm in as_completed(futures_llm):
                futures_llm.result()
            print('end llm')

        return f'用户 {index} finish b function'

    def compute_c(self, index):
        sleep(index)
        return f'用户 {index} finish c function'

    def compute_llm(self, index, i):
        print(f"{ctime()}时刻，用户 {index} 调用第{i}次大模型")
        sleep(5)

    def read_doc(self, index=999):
        print(f'hello, {index}')
        res1 = self.compute_a(index)
        # print(res1)
        res2 = self.compute_b(index)
        # print(res2)
        res3 = self.compute_c(index)
        # print(res3)

        return 'success'


def file_read(index):
    controller = Controller()
    res = controller.read_doc(index=index)
    # print("进程执行成功")
    return f'{res} 运行完成'


def main():
    futures = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        for i in range(10):
            future = executor.submit(file_read, index=i)
            futures.append(future)
        for future in as_completed(futures):
            print(future.result())



# trigger_nums = {
#     4: [1, 2, 3],
#     7: [5, 6],
#     9: [8]
# }
#
# finish_nums = []
#
#
# def need_to_wait(sleep_sec, trigger_nums, finish_nums, condition):
#     res = True
#     if sleep_sec in list(trigger_nums.keys()):
#         for condition_num in list(trigger_nums[sleep_sec]):
#             if condition_num not in finish_nums:
#                 res = False
#     return res
#
#
# def task(sleep_sec=10, condition=None, trigger_nums=None, finish_nums=[], tag='test'):
#     with condition:
#         condition.wait_for(lambda: need_to_wait(sleep_sec, trigger_nums, finish_nums, condition))
#         print(f'[{tag}] start sleep {sleep_sec}, at {ctime()}')
#         sleep(10 - sleep_sec)
#         # print(f'[{tag}] finish sleep {sleep_sec}, at {ctime()}')
#         finish_nums.append(sleep_sec)
#         condition.notify_all()
#     return 100
#
#
# def main():
#     condition = threading.Condition()
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = []
#         for i in range(10):
#             single_process = executor.submit(task, i, condition, trigger_nums, finish_nums, 'TEST')
#             futures.append(single_process)
#
#         for future in futures:
#             if future.done():
#                 print('result is %s' % str(future.result()))
#     # for future in as_completed(futures):
#     #     data = future.result()
#     #     print('result is %s' % str(data))
#
#     executor.shutdown()


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"程序总耗时 {end_time - start_time}")
