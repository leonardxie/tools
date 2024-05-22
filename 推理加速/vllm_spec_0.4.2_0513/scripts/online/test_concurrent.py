#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：vllm_speculative_decoding 
@File    ：test_concurrent.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/25 14:00 
'''
import multiprocessing
import openai
import requests
import logging
import timeit
import concurrent
from concurrent import futures
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ngram_txt = (
#     ("【《水调歌头》\n"
#      "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
#      "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
#      "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
#      "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"
#      "问题: 背诵《水调歌头》全文")
# )

ngram_txt = (
         "《滕王阁序》\n"
         "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
         "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
         "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。\n"
        )

text = [
        ("基于以下【】内的提供的古诗文，回答问题；\n【"
         "《滕王阁序》\n"
         "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
         "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
         "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。\n"
         "】\n"
         "问题: 背诵《滕王阁序》全文")
    ]

# text = [
#     ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
#      "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
#      "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
#      "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
#      "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"
#      "问题: 背诵《水调歌头》全文")
#     ]

data = [
        {
            "role": "system",
            "content": "你是一个背古诗高手，"
        },
        {
            "role": "user",
            "content": text[0]
        }]


def vllm_post_test(pp):
    api_base = "http://10.45.150.84:11003/v1/chat/completions"
    stream = False
    start_time = timeit.default_timer()
    parameters = {
        "max_tokens": 2000,
        "temperature": 0.5,
        "top_p": 0.5,
        "whether_use_speculate": True,
        "ngram_txt": ngram_txt,
    }
    data_input = {
        "stream": stream,
        "messages": data,
        "model": "Qwen-72B-Chat-Int4",
        **parameters,
    }
    completion = requests.post(api_base, json=data_input)
    completion = completion.json()
    res_detail = completion['choices'][0]['message']['content']
    evalTime = timeit.default_timer() - start_time
    logging.info(res_detail)
    print(f"  Evaluation done in total {evalTime} secs")
    print(f" 基于vllm底层实现的投机采样速度：{len(res_detail) / evalTime} tokens/s\n\n")
    return evalTime



def vllm(pp):
    openai.api_base = "http://10.45.150.84:11003/v1"
    openai.api_key = "none"
    stream = False
    start_time = timeit.default_timer()
    completion = openai.ChatCompletion.create(
        # model="Qwen-72B-Chat-Int4",
        model="Qwen-14B-Chat",
        # messages=[{"role": "user", "content": "Hello! What is your name?"}],
        messages=data,
        max_tokens=2000,
        temperature=0.5,
        top_p=0.5,
        whether_use_speculate=True,
        ngram_txt=ngram_txt,
        stream=stream,
    )
    if stream:
        for chunk in completion:
            # print(f"{chunk.choices[0]}")
            try:
                content = chunk.choices[0].delta.content
                if content is None:
                    content = ""
            except Exception as e:
                content = chunk.choices[0].delta.get("content", "")
            print(f"{content}", flush=True, end='')
        print()
        return ""
    else:
        res_detail = completion.choices[0].message.content
        logging.info(res_detail)
        evalTime = timeit.default_timer() - start_time
        print(f"  Evaluation done in total {evalTime} secs")
        print(f" 基于vllm底层实现的投机采样速度：{len(res_detail) / evalTime} tokens/s\n\n")
        return evalTime


def perform_concurrent_test(url, thread_count, request_params):
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        future = executor.submit(vllm_post_test)
        print(future.result())


urls = [
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
    "http://10.45.150.84:11003/v1/chat/completions",
]

def perform_concurrent_test2(thread_count, request_params):
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        future_to_request = {executor.submit(vllm_post_test(url)): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_request):
            # request_params = future_to_request[future]
            try:
                response = future.result()
                print(response)
            except Exception as exc:
                print(f'raised exception {exc}')

url = "http://10.45.150.84:11003/v1/chat/completions"
thread_count = 10
parameters = {
    "max_tokens": 2000,
    "temperature": 0.5,
    "top_p": 0.5,
    "whether_use_speculate": True,
    "ngram_txt": ngram_txt,
}
request_params = {
    "stream": False,
    "messages": data,
    "model": "Qwen-72B-Chat-Int4",
    **parameters,
}
# perform_concurrent_test2(thread_count, request_params)


if __name__ == '__main__':
    total_start = timeit.default_timer()
    all_costs = []
    total_request = 10
    with multiprocessing.Pool(10) as p:
        pool_result = p.map_async(vllm, list(range(total_request)))
        results = pool_result.get(timeout=600000)  # 获取所有进程的返回结果，超时时间为40秒
        for i, result in enumerate(results):
            all_costs.append(result)
        # 关闭进程池
        p.close()
        p.join()
    total_end = timeit.default_timer()
    print(f'total {total_request} times request cost time: {int(total_end - total_start)}s')
    print(f'all_costs: {all_costs}')
    print(f'total {total_request} times request cost time mean: {np.mean(all_costs)}s')