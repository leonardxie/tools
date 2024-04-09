#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：algorithm 
@File    ：test3.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/13 9:17 
'''

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

import requests
import json

from configs.items import ChitChatItem


def parse_stream_helper(line):
    if line:
        return line.decode("utf-8")
    return None


def parse_stream(rbody):
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line

data = {
            'chatType': 'query',
            'query': '',
            'system_message':'请回答user的问题',
            'history':[
                {
                    'role': 'user',
                    'content': '你是谁'
                },
                {
                    'role': 'assistant',
                    'content': '我是小赛'
                },

            ],
            'streaming': True
        }



@app.post("/index")
def index(data_input: ChitChatItem):
    data['query'] = data_input.query
    result = requests.post(url=f'http://10.45.150.84:7507/LLM/request_LLM',
                           json=data, stream=True)

    print(result.text)

    async def ret(dd):
        for chunk in result.iter_lines():
            if chunk:
                content = chunk.decode("utf-8")
            else:
                content = None
            if content is not None:
                data = json.loads(content)
                print(data)
                response_dict = {
                    "errcode": 0,
                    "errmsg": "",
                    "sessionId": dd.sessionId,
                    "conversationId": dd.conversationId,
                    "userId": dd.userId,
                    "type": 2,
                    "stop": data['stop'],
                    "data": {
                        "faqId": "",
                        "contentType": "text",
                        "content": data['content'],
                    }
                }

                yield f"{json.dumps(response_dict, ensure_ascii=False)}\n\n"
            else:
                continue

        # for i in range(50):
        #     str1 = "1111"*i
        #     data = {"1":str1}
        #     # yield进行生成器的数据封装
        #     print(data)
        #     yield json.dumps(data).encode('utf-8')
            # await asyncio.sleep(1)

    return StreamingResponse(ret(data_input))


if __name__ == "__main__":
    uvicorn.run("test3:app", host="0.0.0.0", port=5555)

