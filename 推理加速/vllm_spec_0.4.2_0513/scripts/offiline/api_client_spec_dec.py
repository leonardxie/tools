#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LLMSpeculativeSampling
@File    ：api_client_spec_dec.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/1/22 20:24
'''

"""Example code for running queries from vLLM API server.
Sample Usage:
1. Launch a vLLM server with speculative decoding enabled:
python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 --draft-model TinyLlama/TinyLlama-1.1B-Chat-v0.6 --speculate-length 5
2. Run query using this script:
python api_client_spec_dec.py --prompt "San Francisco is a" --stream
"""

import argparse
import json
from typing import Iterable, List
import timeit
import requests

import logging
import numpy as np
import multiprocessing
import random


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# Sample prompts.
# DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
# TEMPLATE = (
#     "<|im_start|>system\n"
#     "{system_prompt}<|im_end|>\n"
#     "<|im_start|>user\n"
#     "你是一个精通古诗和历史的机器人，\n\n"
#     "Question: {instruction} <|im_end|>\n"
#     "<|im_start|>assistant"
# )

DEFAULT_SYSTEM_PROMPT = """你是一个背古诗高手，"""
TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction} <|im_end|>\n"
    "<|im_start|>assistant\n"
)


def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      max_tokens: int = 256,
                      stream: bool = False,
                      whether_use_speculate: bool = True) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop_token_ids": [151643, 151644, 151645],
        "logits_processors": None,
        # "ngram": True,
        "whether_use_speculate": whether_use_speculate
    }
    response = requests.post(api_url, headers=headers, json=pload)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_content(
            chunk_size=8192,
            decode_unicode=True,
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8")[:-1])
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


def get_api(temperature):
    api_url = f"http://{args.host}:{args.port}/generate"
    max_tokens = 2000
    stream = False
    prompt = [("基于以下【】内的已知内容，回答问题；\n【《滕王阁序》\n"
              "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
              "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
              "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。\n】\n"
              "问题: 背诵《滕王阁序》"),]
              # ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
              #  "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
              #  "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
              #  "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
              #  "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"
              #  "问题: 背诵《水调歌头》全文") ,
              # "讲述一下抗日战争的故事"]
    inputs_list = [generate_prompt(_) for _ in prompt]

    input = random.choice(inputs_list)
    whether_use_speculate = True
    # input = inputs_list[0]
    # print(f"Prompt: {inputs!r}\n", flush=True)
    start_time = timeit.default_timer()
    response = post_http_request(input, api_url, max_tokens, stream, whether_use_speculate)
    if stream:
        num_printed_lines = 0
        char_printed = 0
        for h in get_streaming_response(response):
            line = h[0]
            new_chars = line[char_printed:]
            char_printed = len(line)
            print(f"{new_chars}", flush=True, end='')
            # logging.info(f"{new_chars}")
            num_printed_lines += 1
        print()
    else:
        output = get_response(response)
        line = output[0]
        # ans = line - inputs
        # print(f"{line!r}", flush=True)
        print(f"{line[len(input):]!r}", flush=True)
    evalTime = timeit.default_timer() - start_time
    print(f"  Evaluation done in total {evalTime} secs")
    print(f" 基于vllm底层实现的投机采样速度：{len(line[len(input):]) / evalTime} tokens/s\n\n")
    return evalTime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    # prompt = args.prompt

    # api_url = f"http://{args.host}:{args.port}/generate"
    # max_tokens = 2000
    # # stream = True
    # stream = args.stream
    # 并发测试
    total_start = timeit.default_timer()
    all_costs = []
    total_request = 1
    with multiprocessing.Pool(10) as p:
        pool_result = p.map_async(get_api, list(range(total_request)))
        results = pool_result.get(timeout=6000)  # 获取所有进程的返回结果，超时时间为40秒
        for i, result in enumerate(results):
            all_costs.append(result)
        # 关闭进程池
        p.close()
        p.join()
    total_end = timeit.default_timer()
    print(f'total {total_request} times request cost time: {int(total_end - total_start)}s')
    print(f'all_costs: {all_costs}')
    print(f'total {total_request} times request cost time mean: {np.mean(all_costs)}s')

    # 单数据测试
    # test_api(api_url, max_tokens, stream)
    # response = post_http_request(inputs, api_url, max_tokens, stream)