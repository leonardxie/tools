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

from node import Node
from session import Session
from controller import Controller



def main():
    controller = Controller()
    trigger_nums = {
        1: [],
        2: [],
        3: [],
        4: [1, 2, 3],
        5: [],
        6: [],
        7: [5, 6],
        8: [],
        9: [8]
    }
    node_list = []
    for index, (k, v) in enumerate(trigger_nums.items()):
        children_index =[list(trigger_nums.keys()).index(_) for _ in v]
        tmp_node = Node(index=index, content=k, children=children_index, status=False)
        node_list.append(tmp_node)
    session_list = []
    for i in range(len(node_list)):
        session_node = Session()
        session_node.convert_node_to_session(node_list[i], node_list)
        session_list.append(session_node)

    for i in range(len(session_list)):
        controller.pre_execute_func(session_list[i])
        controller.session_list.append(session_list[i])
        print(f'时间{ctime()}, session {session_list[i].session_id} 完成数据前处理，放入list_waiting')
        controller.list_waiting.append(session_list[i])

    sleep(1000)
    sleep(1000)



    print(session_list)



if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"程序总耗时 {end_time - start_time}")
