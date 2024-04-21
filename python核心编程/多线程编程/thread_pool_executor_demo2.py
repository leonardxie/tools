#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：thread_pool_executor_demo2.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/21 20:30 
'''

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# 参数times用来模拟网络请求的时间
def get_html(times):
    time.sleep(times)
    print("get page {}s finished".format(times))
    return times


executor = ThreadPoolExecutor(max_workers=2)
urls = [3, 2, 4]  # 并不是真的url
all_task = [executor.submit(get_html, (url)) for url in urls]

for future in as_completed(all_task):
    data = future.result()
    print("in main: get page {}s success".format(data))

# 执行结果
'''
get page 2s finished
in main: get page 2s success
get page 3s finished
in main: get page 3s success
get page 4s finished
in main: get page 4s success
'''

"""
as_completed(提交的任务)
上面虽然提供了判断任务是否结束的方法，但是不能在主线程中一直判断。
有时候我们是得知某个任务结束了，就去获取结果，而不是一直判断每个任务有没有结束。
可以使用as_completed方法一次取出所有任务的结果
as_comp1eted()方法是一个生成器，在没有任务完成的时候，会阻塞，
在有某个任务完成的时候，会yie1d这个任务，就能执行for循环下面的语句，
然后继续阻塞住，循环到所有的任务结束。从结果也可以看出，先完成的任务会先通知主线程
"""