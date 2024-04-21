#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_threading_demo.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 14:32 
'''

# 使用threading实现多线程的操作
import threading
import time


# 定义统计的数量
class Sum:
    count = 0


# 使用三个线程执行当前的Sum.count++操作的时候就出现了线程安全问题，注意这里的target=sum,如果target=sum()就会让main线程执行
# 所以这个里需要加锁
def sum():
    while Sum.count < 10:
        time.sleep(1)
        Sum.count += 1
        current_name = threading.currentThread().getName()
        # 获取当前执行线程的名称
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 获取当前执行的时间
        print("{0}:当前执行操作的时间：{1},当前count的结果为：{2}".format(current_name, current_time, Sum.count))


def main():
    print("线程开始执行。。。")
    threading.Thread(target=sum, name="线程一").start()
    threading.Thread(target=sum, name="线程二").start()
    threading.Thread(target=sum, name="线程三").start()
    print("线程结束执行。。。")


if __name__ == '__main__':
    main()
