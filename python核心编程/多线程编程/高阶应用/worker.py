#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：worker.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/22 14:41 
'''
from time import ctime, sleep
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from session import Session


class Worker:
    def __init__(self):
        pass

    def task(self, sleep_sec=10, tag='test'):
        print(f'[{tag}] start sleep {sleep_sec}, at {ctime()}')
        sleep(10 - sleep_sec)
        # print(f'[{tag}] finish sleep {sleep_sec}, at {ctime()}')

        return sleep_sec

    def compute_a(self, single_session: Session):
        index = single_session.session_id
        sleep(1)

    def compute_b(self, index):
        # print( f"用户 {index} 调用大模型")
        # 现有方案：串行调用大模型
        for i in range(1, 11):
            self.compute_llm(index, i)  # 10个用户，并发运行10次，每个用户调10次大模型，总耗时 126s
        # 优化方案2：多线程调用大模型 # 10个用户，并发运行10次，每个用户并发调10次大模型，总耗时 46s
        # futures_llm = []
        # with ThreadPoolExecutor(max_workers=5) as executor_tmp:
        #     print('start llm')
        #     for i in range(1, 11):
        #         futures_llm.append(executor_tmp.submit(self.compute_llm, index, i))
        #     for futures_llm in as_completed(futures_llm):
        #         futures_llm.result()
        #     print('end llm')
        print(f'用户 {index} finish b function')
        return 200

    def compute_c(self, single_session: Session):
        index = single_session.session_id
        sleep(3)
        print(f'时间{ctime()}, session {index} 完成后处理函数')
        return 200

    def compute_llm(self, single_session: Session):
        index = single_session.session_id
        print(f"时间{ctime()}, session 用户 {index} 调用大模型")
        sleep(10)

    def read_doc(self, tmp_session: Session):
        index = tmp_session.session_id
        print(f'hello, {index}')
        res1 = self.compute_a(index)
        # print(res1)
        res2 = self.compute_b(index)
        # print(res2)
        res3 = self.compute_c(index)
        # print(res3)

        return 'success'


def main():
    worker = Worker()
    print(worker.task(5))


if __name__ == '__main__':
    main()
