#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：myThread.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 15:32 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_threading_demo3.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 15:09 
'''
import threading
from time import sleep, ctime

loops = [4, 2]

class MyThread(threading.Thread):
    def __init__(self, func, args, name = ''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def getResult(self):
        return self.res

    def run(self):
        print(f"starting {self.name} at: {ctime()}")
        self.res= self.func(*self.args)
        print(f"{self.name} finished at: {ctime()}")

def loop(nloop, nsec):
    print(f'start loop {nloop} at: {ctime}')
    sleep(nsec)
    print(f'loop {nloop} done at: {ctime()}')

def main():
    print(f'starting at: {ctime()}')
    threads = []
    nloops = range(len(loops))

    for i in nloops: # create all threads
        t = MyThread(loop, (i, loops[i]), loop.__name__)
        threads.append(t)

    for i in nloops: # start all threads
        threads[i].start()

    for i in nloops: # wait for all completion
        threads[i].join()

    print(f'all DONE at: {ctime()}')

if __name__ == '__main__':
    main()