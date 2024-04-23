#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：controller.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024-04-23 11:31 
'''

from time import sleep, ctime
from typing import List
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from concurrent.futures._base import Future
from session import Session
from worker import Worker


class Controller:
    def __init__(self):
        self.session_list: List[Session] = []
        self.list_waiting: List[Session] = []
        self.list_executing: List[Session] = []
        self.list_finish: List[Session] = []

        self.worker = Worker()
        self.pre_execute_func = self.worker.compute_a
        self.post_execute_func = self.worker.compute_c
        self.execute_func = self.worker.compute_llm
        self.executor = ProcessPoolExecutor(max_workers=5)

        self.worker_thread = threading.Thread(target=self.cache_consumer, daemon=True)
        self.worker_thread.start()

    def cache_consumer(self):
        print(f"时间{ctime()}, check_consume")
        while True:
            sleep(10)
            if not self.list_waiting and not self.list_executing and not self.list_finish:
                continue
            print(f'时间{ctime()}, {len(self.list_waiting)}个session等待submit;{len(self.list_executing)}个session在执行;{len(self.list_finish)}个session执行结束;\n')

            if self.list_waiting:
                self.check_waiting_list()

            if self.list_executing:
                self.check_execute_list()

            if self.list_finish:
                self.check_finish_list()

    def session_execute(self, single_session: Session):
        future = self.executor.submit(self.execute_func, single_session)
        return {'session': single_session, 'future': future}

    def check_waiting_list(self):
        waiting_list = []
        for session_tmp in self.list_waiting:
            def all_true(lst):
                return all(lst)
            if not session_tmp.children or all_true(session_tmp.child_is_done):
                res = self.session_execute(session_tmp)
                self.list_executing.append(res)
            else:
                waiting_list.append(session_tmp)
                continue
        self.list_waiting = waiting_list

    def check_execute_list(self):
        execute_list = []
        for session_dict in self.list_executing:
            future: Future = session_dict.get('future')
            if not future.done():
                execute_list.append(session_dict)
                continue
            else:
                self.list_finish.append(session_dict['session'])
                print(f"时间{ctime()}, session {session_dict['session'].session_id} 完成请求大模型，放入list_finish")
                continue

        self.list_executing = execute_list

    def check_finish_list(self):
        for single_session in self.list_finish:
            if single_session.is_done:
                continue
            self.post_execute_func(single_session)
            single_session.is_done = True
            for session_tmp in self.session_list:
                if single_session.session_id in session_tmp.children:
                    ind = session_tmp.children.index(single_session.session_id)
                    session_tmp.child_is_done[ind] = True
