#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：construct_concurrent_pool.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 10:38 
'''

"""
此脚本用来测试并发执行单个函数
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import Future
from typing import Callable


class Controller:

    def __init__(
            self,
            excute_func: Callable = None,
            finish_func: Callable = session_finish_return
    ):
        self.Worker = FileWorker

        # 确定submit、execute函数
        self.execute_func = execute_func or self.Worker.load_split_add
        self.finish_func = finish_func

        # 创建一个进程池执行器
        self.executor = ProcessPoolExecutor(max_workers=5)

        # 创建一个守护线程来处理队列中的数据
        self.worker_thread = threading.Thread(tartget=self.cache_consumer, daemon=True)
        self.worker_thread.start()

    def session_execute(self, session: SingleSession):
        """
             处理数据主任务
         """
        future = self.executor.submit(self.execute_func, session)

    def cache_consumer(self):

