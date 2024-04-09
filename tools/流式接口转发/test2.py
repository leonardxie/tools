#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：algorithm 
@File    ：test2.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/13 9:17 
'''


import asyncio
import json

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()


@app.post("/index")
async def index():

    async def ret():
        url = 'http://localhost:5555/index'
        response = requests.get(url, stream=True)
        print(response)
        for chunk in response.iter_content(chunk_size=1024,
                                           decode_unicode=False,
                                           ):
            print(chunk)

            yield chunk
            # await asyncio.sleep(1)

    return StreamingResponse(ret())




if __name__ == "__main__":
    uvicorn.run("test2:app", host="0.0.0.0", port=5556)
