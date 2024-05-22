#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：vllm_speculative_decoding 
@File    ：vllm_spec_speed.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/2/28 17:34 
'''

import pickle
from json import JSONDecodeError

import requests
import logging
import time
import json
# vllm的测试速度
import openai
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import timeit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ngram_txt = ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
     "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
     "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
     "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
     "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n")
# ngram_txt = (
#          "《滕王阁序》\n"
#          "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
#          "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
#          "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。\n"
#          # "《蜀道难》\n"
#          # "噫吁戏，危乎高哉！蜀道之难，难于上青天！蚕丛及鱼凫，开国何茫然！尔来四万八千岁，不与秦塞通人烟。西当太白有鸟道，可以横绝峨眉巅。地崩山摧壮士死，然后天梯石栈相钩连。上有六龙回日之高标，下有冲波逆折之回川。黄鹤之飞尚不得过，猿猱欲度愁攀援。青泥何盘盘，百步九折萦岩峦。扪参历井仰胁息，以手抚膺坐长叹。 问君西游何时还？畏途巉岩不可攀。但见悲鸟号古木，雄飞雌从绕林间。又闻子规啼夜月，愁空山。蜀道之难，难于上青天，使人听此凋朱颜！连峰去天不盈尺，枯松倒挂倚绝壁。飞湍瀑流争喧豗，砯崖转石万壑雷。其险也如此，嗟尔远道之人胡为乎来哉！ 剑阁峥嵘而崔嵬，一夫当关，万夫莫开。所守或匪亲，化为狼与豺。朝避猛虎，夕避长蛇；磨牙吮血，杀人如麻。锦城虽云乐，不如早还家。蜀道之难，难于上青天，侧身西望长咨嗟！\n"
#          # "《兰亭集序》\n"
#          # "永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭，修禊事也。群贤毕至，少长咸集。此地有崇山峻岭，茂林修竹，又有清流激湍，映带左右，引以为流觞曲水，列坐其次。虽无丝竹管弦之盛，一觞一咏，亦足以畅叙幽情。 是日也，天朗气清，惠风和畅。仰观宇宙之大，俯察品类之盛，所以游目骋怀，足以极视听之娱，信可乐也。 夫人之相与，俯仰一世。或取诸怀抱，悟言一室之内；或因寄所托，放浪形骸之外。虽趣舍万殊，静躁不同，当其欣于所遇，暂得于己，快然自足，不知老之将至；及其所之既倦，情随事迁，感慨系之矣。向之所欣，俯仰之间，已为陈迹，犹不能不以之兴怀，况修短随化，终期于尽！古人云：“死生亦大矣。”岂不痛哉！ 每览昔人兴感之由，若合一契，未尝不临文嗟悼，不能喻之于怀。固知一死生为虚诞，齐彭殇为妄作。后之视今，亦犹今之视昔，悲夫！故列叙时人，录其所述，虽世殊事异，所以兴怀，其致一也。后之览者，亦将有感于斯文。\n"
#          # "《阿房宫赋》\n"
#          # "六王毕，四海一，蜀山兀，阿房出。覆压三百余里，隔离天日。骊山北构而西折，直走咸阳。二川溶溶，流入宫墙。五步一楼，十步一阁；廊腰缦回，檐牙高啄；各抱地势，钩心斗角。盘盘焉，囷囷焉，蜂房水涡，矗不知其几千万落。长桥卧波，未云何龙？复道行空，不霁何虹？高低冥迷，不知西东。歌台暖响，春光融融；舞殿冷袖，风雨凄凄。一日之内，一宫之间，而气候不齐。 妃嫔媵嫱，王子皇孙，辞楼下殿，辇来于秦。朝歌夜弦，为秦宫人。明星荧荧，开妆镜也；绿云扰扰，梳晓鬟也；渭流涨腻，弃脂水也；烟斜雾横，焚椒兰也。雷霆乍惊，宫车过也；辘辘远听，杳不知其所之也。一肌一容，尽态极妍，缦立远视，而望幸焉；有不得见者，三十六年。燕赵之收藏，韩魏之经营，齐楚之精英，几世几年，剽掠其人，倚叠如山；一旦不能有，输来其间。鼎铛玉石，金块珠砾，弃掷逦迤，秦人视之，亦不甚惜。 嗟乎！一人之心，千万人之心也。秦爱纷奢，人亦念其家。奈何取之尽锱铢，用之如泥沙？使负栋之柱，多于南亩之农夫；架梁之椽，多于机上之工女；钉头磷磷，多于在庾之粟粒；瓦缝参差，多于周身之帛缕；直栏横槛，多于九土之城郭；管弦呕哑，多于市人之言语。使天下之人，不敢言而敢怒。独夫之心，日益骄固。戍卒叫，函谷举，楚人一炬，可怜焦土！ 灭六国者六国也，非秦也；族秦者秦也，非天下也。嗟乎！使六国各爱其人，则足以拒秦；使秦复爱六国之人，则递三世可至万世而为君，谁得而族灭也？秦人不暇自哀，而后人哀之；后人哀之而不鉴之，亦使后人而复哀后人也。"
#         )

text = [
        # ("背诵《滕王阁序》全文")
        ("背诵《水调歌头》全文")
    ]
system = [
    # ("基于以下【】内的提供的古诗文，回答问题；\n【"
    #  "《滕王阁序》\n"
    #  "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
    #  "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
    #  "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。\n"
    #  "】\n")
    ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
     "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
     "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
     "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
     "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"),
]
data = [
        {
            "role": "system",
            "content": f"你是一个背古诗高手，{system[0]}"
        },
        {
            "role": "user",
            # "content": "背诵滕王阁序"
            "content": text[0]
        }]


def vllm(pp):
    openai.api_base = "http://10.45.150.84:11003/v1"
    openai.api_key = "none"
    stream = False
    start_time = timeit.default_timer()
    completion = openai.ChatCompletion.create(
        model="Qwen-72B-Chat-Int4",
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
        evalTime = timeit.default_timer() - start_time
        logging.info(res_detail)
        print(f"  Evaluation done in total {evalTime} secs")
        print(f" 基于vllm底层实现的投机采样速度：{len(res_detail) / evalTime} tokens/s\n\n")
        return evalTime


def vllm_stream(data):
    openai.api_base = "http://10.45.150.84:11003/v1"
    openai.api_key = "none"
    stream = True
    # print(len(text[0]))
    completion = openai.ChatCompletion.create(
        model="Qwen-72B-Chat-Int4",
        # messages=[{"role": "user", "content": "Hello! What is your name?"}],
        messages=data,
        max_tokens=2000,
        temperature=0.5,
        top_p=0.5,
        whether_use_speculate=True,
        ngram_txt=ngram_txt,
        stream=stream,
    )
    for chunk in completion:
        try:
            # print(chunk.choices[0])
            content = chunk.choices[0].delta.content
            if content is None:
                content = ""
        except Exception as e:
            content = chunk.choices[0].delta.get("content", "")
        if chunk.choices[0].get("finish_reason") and chunk.choices[0].finish_reason == "stop":
            stop = True
        else:
            stop = False
        res = {
            "errcode": 0,
            "errmsg": "",
            "sessionId": "abc",  # 会话标识id, 与input对齐
            "conversationId": "456",  # 单轮返回标识id，与input不同
            "userId": "xxx",  # 用户标识id
            "type": 2,  # 1：faq，2：llm
            "stop": stop,  # 流式输出标志，True为停止，True时字符为空 True / False
            "data": {
                "faqId": "",
                "contentType": "text",
                "content": content,
            }
        }

        yield res


def parse_stream(rbody):
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


def parse_stream_helper(line):
    if line:
        if line.strip() == b"data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: "):]
            return line.decode("utf-8")
        else:
            return None
    return None


def _interpret_response_line(rbody: str, rcode: int, rheaders):
    # HTTP 204 response code does not have any content in the body.
    if rcode == 204:
        return None, rheaders

    try:
        if 'text/plain' in rheaders.get('Content-Type', ''):
            data = rbody
        else:
            data = json.loads(rbody)
    except (JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f'LLM: error in spliter request llm: {e}')
    return data, rheaders


def vllm_post_stream_test():
    api_base = "http://10.45.139.204:11003/v1/chat/completions"
    stream = True
    output = ''
    parameters = {
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.5,
        "whether_use_speculate": True,
        "ngram_txt": ngram_txt,
    }
    data_input = {
        "stream": stream,
        "messages": data,
        "model": "Qwen1.5-32B-Chat",
        **parameters,
    }
    completion = requests.post(api_base, json=data_input, stream=stream)
    for chunk in parse_stream(completion.iter_lines()):
        chunk, headers = _interpret_response_line(chunk, completion.status_code, completion.headers)
        try:
            content = chunk["choices"][0]["delta"].get("content", "")
            if content is None:
                content = ""
        except Exception as e:
            content = chunk.choices[0].delta.get("content", "")

        print(content, flush=True, end='')


def vllm_post_test(pp):
    api_base = "http://10.45.139.204:11003/v1/chat/completions"
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
        "model": "Qwen1.5-32B-Chat",
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


if __name__ == "__main__":
    # 非流式测试
    res = vllm_post_test('ss')

    # 流式测试
    vllm_post_stream_test()

    # 并发测试
    total_start = timeit.default_timer()
    all_costs = []
    total_request = 5

    with multiprocessing.Pool(10) as p:
        pool_result = p.map_async(vllm_post_test, list(range(total_request)))
        results = pool_result.get(timeout=600)  # 获取所有进程的返回结果，超时时间为40秒
        for i, result in enumerate(results):
            all_costs.append(result)
        # 关闭进程池
        p.close()
        p.join()
    total_end = timeit.default_timer()
    print(f'total {total_request} times request cost time: {int(total_end - total_start)}s')
    print(f'all_costs: {all_costs}')
    print(f'total {total_request} times request cost time mean: {np.mean(all_costs)}s')



