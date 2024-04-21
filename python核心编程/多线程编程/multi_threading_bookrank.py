#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools
@File    ：multi_threading_bookrank.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/4/19 16:36 
'''

from atexit import register
from re import compile
from threading import Thread

from time import ctime, sleep
# from urllib.request import urlopen as uopen

from msedge.selenium_tools import Edge, EdgeOptions



edge_options = EdgeOptions()
edge_options.use_chromium = True #初始化Edge
edge_options.headless = True #静默运行，网页不显示

EDGE_DRIVER_PATH = r"F:\Git_Local_Repository\tools\python核心编程\多线程编程\edgedriver_win64\msedgedriver.exe"
BROWSER = Edge(executable_path=EDGE_DRIVER_PATH, options=edge_options)


REGEX = compile("#([\d,]+) in Books ")
AMZN = 'https://www.amazon.com/dp/'
ISBNs = {
    '0132269937': 'Core Python Programming',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals',
}
# 2023.07.05, Amazon进行了反爬虫限制，无法在程序中通过简单的get请求获取页面内容
#
def getRanking(isbn):
    # page = uopen(AMZN +isbn) # or str.format()
    # data = page.read()
    # page.close()
    # post请求方式
    # page = requests.get(f"{AMZN}{isbn}",headers=headers)

    BROWSER.get(url=f"{AMZN}{isbn}")
    return REGEX.findall(BROWSER.page_source)[0]

def _showRanking(isbn):
    print(f"正在处理isbn号为 {isbn} 的书")
    print(f"- {ISBNs[isbn]}的排名为{getRanking(isbn)}名")
    print(f" isbn号为 {isbn} 的书处理完成 at {ctime()}")

def main():
    print(f'At {ctime()}, 在亚马逊官网...')
    threads = []
    for isbn in ISBNs:
        # 串行
        # _showRanking(isbn)
        # 并行
        t = Thread(target=_showRanking, args=(isbn,), name=isbn)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()


@register
def _atexit():
    print(f'all DONE at {ctime()}')


if __name__ == '__main__':
    main()

