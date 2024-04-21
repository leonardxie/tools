#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：thread_pool_executor_demo.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/21 20:21 
'''

from concurrent.futures import ThreadPoolExecutor
import time


# 参数times用来模拟网络请求的时间
def get_html(times):
    time.sleep(times)
    print("get page {}s finished".format(times))
    return times


executor = ThreadPoolExecutor(max_workers=2)
# 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
task1 = executor.submit(get_html, (3))
task2 = executor.submit(get_html, (2))
# done方法用于判定某个任务是否完成
print(task1.done())
# cancel方法用于取消某个任务,该任务没有放入线程池中才能取消成功
print(task2.cancel())
time.sleep(4)
print(task1.done())
# result方法可以获取task的执行结果
print(task1.result())

# 执行结果
'''
False  # 表明task1未执行完成
False  # 表明task2取消失败，因为已经放入了线程池中
get page 2s finished
get page 3s finished
True  # 由于在get page 3s finished之后才打印，所以此时task1必然完成了
3     # 得到task1的任务返回值

'''

"""
ThreadPoolExecutor构造实例的时候，传入max_workers参数来设置线程池中最多能同时运行的线程数目
使用 submit() 函数来提交线程需要执行的任务(函数名和参数)到线程池中，并返回该任务的句柄(类似于文件、画图)，
注意submit()不是阻塞的，而是立即返回
通过submit() 函数返回的任务句柄，能够使用done()方法判断该任务是否结束
使用cancel() 方法可以取消提交的任务，如果任务已经在线程池中运行了，就取消不了。

这个例子中，线程池的大小设置为2，任务已经在运行了，所以取消失败。
如果改变线程池的大小为1，那么先提交的是task1，task2还在排队等候，这是时候就可以成功取消
使用result() 方法可以获取任务的返回值。查看内部代码，发现这个方法是阻塞的
"""