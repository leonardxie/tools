#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LLMSpeculativeSampling
@File    ：offline_inference_spec_dec_with_ngram.py
@IDE     ：PyCharm
@Author  ：解书贵
@Date    ：2024/1/22 20:24
'''
"""Containing tests that check for regressions in vLLM's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
from vllm import LLM, SamplingParams
import timeit
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Sample prompts.
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个精通古诗和历史的机器人，\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: {instruction} <|im_end|>\n"
    "<|im_start|>assistant"
)


def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})

question = [
    # "背诵《滕王阁序》",
    "背诵《水调歌头》"
]
prompts = [
    # ("基于以下【】内的已知内容，回答问题；\n【《滕王阁序》\n"
    #  "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
    #  "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
    #  "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。】\n"),
    # ("基于以下【】内的已知内容，回答问题；\n【《滕王阁序》\n"
    #  "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯。\n"
    #  "时维九月，序属三秋。潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。临帝子之长洲，得天人之旧馆。层峦耸翠，上出重霄；飞阁流丹，下临无地。鹤汀凫渚，穷岛屿之萦回；桂殿兰宫，即冈峦之体势。披绣闼，俯雕甍，山原旷其盈视，川泽纡其骇瞩。闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之舳。云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨，雁阵惊寒，声断衡阳之浦。遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。望长安于日下，目吴会于云间。地势极而南溟深，天柱高而北辰远。关山难越，谁悲失路之人？萍水相逢，尽是他乡之客。怀帝阍而不见，奉宣室以何年？\n"
    #  "嗟乎！时运不齐，命途多舛。冯唐易老，李广难封。屈贾谊于长沙，非无圣主；窜梁鸿于海曲，岂乏明时？所赖君子见机，达人知命。老当益壮，宁移白首之心？穷且益坚，不坠青云之志。酌贪泉而觉爽，处涸辙以犹欢。北海虽赊，扶摇可接；东隅已逝，桑榆非晚。孟尝高洁，空余报国之情；阮籍猖狂，岂效穷途之哭！勃，三尺微命，一介书生。无路请缨，等终军之弱冠；有怀投笔，慕宗悫之长风。舍簪笏于百龄，奉晨昏于万里。非谢家之宝树，接孟氏之芳邻。他日趋庭，叨陪鲤对；今兹捧袂，喜托龙门。杨意不逢，抚凌云而自惜；钟期既遇，奏流水以何惭？呜乎！胜地不常，盛筵难再；兰亭已矣，梓泽丘墟。临别赠言，幸承恩于伟饯；登高作赋，是所望于群公。敢竭鄙怀，恭疏短引；一言均赋，四韵俱成。请洒潘江，各倾陆海云尔。】\n"
    #  "问题: 背诵《滕王阁序》全文"),
    ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
     "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
     "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
     "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
     "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"),
    # "简单讲述一下抗日战争的历史",
    # ("基于以下【】内的已知内容，回答问题；\n【《水调歌头》\n"
    #  "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。\n"
    #  "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。 \n"
    #  "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？\n"
    #  "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。】\n"
    #  "问题: 背诵《水调歌头》全文"),
    # "简单讲述一下抗日战争的历史",
]

inputs = [generate_prompt(question[i], system_prompt=text) for i, text in enumerate(prompts)]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.5,
                                 max_tokens=5000,
                                 stop_token_ids=[151643, 151645],
                                 logits_processors=None,
                                 ngram=None,
                                 whether_use_speculate=False,
                                 ngram_text=prompts[0])

# Create an LLM.
# llm = LLM(model="lmsys/vicuna-13b-v1.5",
#           draft_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
#           speculate_length=5)
llm = LLM(model="/data01/01model_hub/LLM/Qwen1.5-14B-Chat",
          gpu_memory_utilization=0.9,
          trust_remote_code=True,
          dtype=torch.float16,
          disable_log_stats=False,
          # 投机采样参数
          # speculative_model="/data01/01model_hub/LLM/Qwen1.5-1_8B-Chat",
          speculative_model="[ngram]",
          num_speculative_tokens=5,
          speculative_max_model_len=32768,
          speculative_disable_by_batch_size=32,
          ngram_prompt_lookup_max=100,
          ngram_prompt_lookup_min=10,
          use_v2_block_manager=True,
          )

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time = timeit.default_timer()

outputs = llm.generate(inputs, sampling_params)
evalTime = timeit.default_timer() - start_time

for output in outputs:
    # prompt = output.prompt
    # logging.info(f"投机采样结果:\n\n {output.outputs[0].text}")
    logging.info(f"  Evaluation done in total {evalTime} secs\n")
    # logging.info(f"  Prompt: {prompt!r}\n")
    logging.info(f" speed: {len(output.outputs[0].text) / evalTime} tokens/s")
    # mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    # logging.info(f" 平均接受长度：{mean_num_accepted}")

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
#     print(
#         f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
#     )
