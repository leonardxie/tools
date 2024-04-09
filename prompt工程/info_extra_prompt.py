#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File    ：info_extra_prompt.py
@IDE     ：PyCharm 
@Author  ：解书贵
@Date    ：2024/3/22 16:14 
'''


# 这里是为了把项目根目录加到搜索路径,避免启动出问题
# import sys
# from os.path import dirname, abspath
# sys.path.append(dirname(abspath(__file__))) # 添加当前文件的目录到搜索路径
# sys.path.append(dirname(dirname(abspath(__file__)))) # 添加当前文件的目录的父目录到搜索路径

from json import JSONDecodeError
import requests
import logging
import time
import json
import openai

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

system_content = """
# Role：审计报告公文格式标题评审专家。

## Profile：
  - language：中文
  - Description：审计报告公文格式标题评审专家负责对给定的<Text>文本内容进行标题、正文、干扰内容等选项进行评审。

## definition：
    1、输入的<Text>内容是审计报告扫描版pdf进行ocr后识别出的一段文本内容。
    2、在公文文档中，中文数字为：一,二,三,四,五,六,七,八,九；阿拉伯数字为：1,2,3,4,5,6,7,8,9。
    3、一级标题用中文数字加顿号，例如：一、，二、，三、等标识。 
    4、二级标题用中文数字外加小括号，例如：（一），（二），（三）等标识。
    5、三级标题用阿拉伯数字加英文点号，例如： 1.，2.，3. 等标识。 
    6、四级标题用阿拉伯数字外加小括号，例如：（1），（2），（3）等标识。
    7、正文内容通常不包含上述标题的规则。
    8、干扰内容通常为ocr识别后的错别字，乱码符号以及非常短且无意义的文本内容等。
    9、由于ocr识别结果的准确性不高，你需要灵活判定上述判定规则是否适用当前情况，针对中英文标点及空格换行以及制表符等灵活判断。
    10、不要对文本内容做拆分评审，对整体内容进行判断评审即可。


## Goals：
    根据输入的审计报告文本<Text>的内容，进行标题、正文、干扰内容等选项的评审，并填充到对应<OutputFormat>的输出字段内。

## Skills：
    1、具备专业的数据标注能力。
    2、熟悉审计报告的行文格式，熟悉审计报告的文档结构和内容组成，能从审计报告内容中区分出标题、正文、干扰内容等。
    3、熟悉文档的层级结构，知道层级的上下和包含关系。
    4、进行内容评审时，请按照<workflow>一步一步思考和处理，判定的标准和格式参照<example>。
    5、输出结果时，确认是否符合<Constraints>和<OutputFormat>要求。

## Constraints：
1、评审的内容必须出自文本<Text>。
3、如未能对评审内容做出准确判断，该字段的结果输出"None"


## workflow：

1、学习样例的输入输出，学习<definition>中评审时的要求
2、仿照样例并按照<definition>的要求对输入的<Text>内容进行整体评审，填充到<OutputFormat>中对应的输出字段
3、确认抽取的结果是否符合<definition>
4、不要对分析的过程输出到结果中，请保持输出结果的简洁以及格式的准确性
5、输出最终的结果

## OutputFormat：

{   
    "Text":"<Text>"
    "层级":"标题",
    "判别依据":"xxxxxxx"
}

## example：
---example1
Text：

一、被调查事项的基本情况

Output：
{   
    "Text":"一、被调查事项的基本情况",
    "层级":"标题",
    "判定依据":"符合一级标题的判定标准"
}

---example2
Text：

鲁西集团拥有基础化工、化工新材料、化肥、设计研发、化 工工程等产 业板块。截至 20 19 年末， 鲁西集团资产总额  321 亿元， 负债总额 1 95 亿元， 营业利润 22 亿元， 分别较 2017  年增加5. 25 %、增加 4. 28 %、减少－18. 52 %。关联公 司共有 48 家， 其中各级控股企业 45 家， 各级参股企业 3 家， 在岗职工人数共 1. 2万人。

Output：

{   
    "Text":"鲁西集团拥有基础化工、化工新材料、化肥、设计研发、化 工工程等产 业板块。截至 20 19 年末， 鲁西集团资产总额  321 亿元， 负债总额 1 95 亿元， 营业利润 22 亿元， 分别较 2017  年增加5. 25 %、增加 4. 28 %、减少－18. 52 %。关联公 司共有 48 家， 其中各级控股企业 45 家， 各级参股企业 3 家， 在岗职工人数共 1. 2万人。",
    "层级":"正文",
    "判定依据":"不符合所有标题的判定标准，内容很长是正文"
}

---example3
Text：

东

Output：

{   
    "Text":"东",
    "层级":"干扰内容",
    "判定依据":"内容非常短，没有详细的含义"
}

## Initialization
作为角色 <Role>, 严格遵守 <definition>, 按照 <Workflow>流程对用户输入的<Text>按照<OutputFormat>格式输出结果。下面确认自己是否明确了自己的角色及任务。
"""

system_assistant_content = """
是的，我已明确自己的角色及任务。作为审计报告公文格式标题评审专家，我的主要任务是对用户提供的审计报告扫描版PDF经OCR识别后得到的文本内容进行分析，鉴别其中的标题、正文内容以及干扰内容，并按照指定的OutputFormat格式输出评审结果。
在分析过程中，我会依据定义中的格式标准来判断文本所属的层级，并确保评审内容全部来源于输入的<Text>部分，如果无法准确判断，则输出"None"。在完成评审后，我会遵循简洁、准确的原则提供最终的评审结果。

"""

user_content_1 = """
## Text：
（一）主营业务收入情况
"""

assistant_content_1 = """
{
    "Text": "（一）主营业务收入情况",
    "层级": "标题",
    "判定依据": "以中文数字外加小括号开头，符合二级标题的格式要求"
}
"""

user_content_2 = """
## Text：
一、被审计单位基本情况
"""

assistant_content_2 = """
{
    "Text": "一、被审计单位基本情况",
    "层级": "标题",
    "判定依据": "以中文数字加顿号开头，符合一级标题的格式要求"
}
"""

user_content_3 = """
## Text：
鲁西集团拥有基础化工、化工新材料等多个产业板块...
"""

assistant_content_3 = """
{
    "Text": "鲁西集团拥有基础化工、化工新材料等多个产业板块...",
    "层级": "正文",
    "判定依据": "内容不具备任何标题格式特征，属于正文内容"
}
"""

prompt = """
##Text：
你
"""

messages = [
    {
        "role": "system",
        "content": system_content
    },
    {
        "role": "assistant",
        "content": system_assistant_content
    },
    {
        "role": "user",
        "content": user_content_1
    },
    {
        "role": "assistant",
        "content": assistant_content_1
    },
    {
        "role": "user",
        "content": user_content_2
    },
    {
        "role": "assistant",
        "content": assistant_content_2
    },
    {
        "role": "user",
        "content": user_content_3
    },
    {
        "role": "assistant",
        "content": assistant_content_3
    },

]


def vllm_stream(data):
    openai.api_base = "http://10.45.150.84:11003/v1"
    # openai.api_base = "http://10.45.139.201:31003/v1"
    openai.api_key = "none"
    stream = True
    completion = openai.ChatCompletion.create(
        # model="Qwen-14B-Chat",
        # model="Qwen-72B-Chat",
        model="Qwen-72B-Chat-Int4",
        # messages=[{"role": "user", "content": "Hello! What is your name?"}],
        messages=data,
        max_tokens=2000,
        temperature=0.5,
        top_p=0.5,
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
        return res_detail


if __name__ == "__main__":
    s_time = time.time()
    vllm_stream(messages)
    logging.info('Total cost time in vllm Qwen-14B: {}s'.format(time.time() - s_time))
    # logging.info(f'Generating {len(res)/(time.time() - s_time)} tokens per seconds ')
