# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------
# @Time   : 2023/7/25 21:45
# @Author : Jason Wang
# -----------------------------

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站
import argparse
import time
import shutil
from huggingface_hub import snapshot_download


def get_save_dir(symlink_path, save_rt_dir):
    symlink_path_split_res = symlink_path.split('/')
    sub_save_dir = '/'.join(symlink_path_split_res[symlink_path_split_res.index("snapshots") + 2: -1])
    save_dir = os.path.join(save_rt_dir, sub_save_dir)
    if not os.path.exists(save_dir):
        cmd = f"mkdir -p {save_dir}"
        os.system(cmd)  # 创建下载模型目录中的子目录

    return save_dir


def move_cache_file(snapshot_folder, save_rt_dir):
    for rt, folders, snapshots in os.walk(snapshot_folder):
        for snapshot in snapshots:
            snapshot_path = os.path.join(rt, snapshot)
            cache_path = os.path.realpath(snapshot_path)  # 获取软连接所指缓存文件的实际路径
            save_dir = get_save_dir(symlink_path=snapshot_path, save_rt_dir=save_rt_dir)
            save_path = os.path.join(save_dir, snapshot)  # link name 为最终文件名
            shutil.move(src=cache_path, dst=save_path)  # 将cache file移动到save dir
            #shutil.copy(src=cache_path, dst=save_path)
            #cmd = f"cp '{cache_path}' '{save_path}'"
            #os.system(cmd)  # 将cache file移动到save dir


def download_huggingface_model(repo_id, save_dir=""):
    model_name = repo_id.split('/')[-1]
    cache_dir = os.path.join(save_dir, model_name + "_cache")
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    while True:
        try:
            snapshot_folder = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                # resume_download=True,
            )
            print(f"Moving `{model_name}` cache file to `{save_dir}` ...")
            move_cache_file(snapshot_folder=snapshot_folder, save_rt_dir=save_dir)
            print(f"Move `{model_name}` cache file successfully!")
            print(f"Deleting `{model_name}` cache_dir: `{cache_dir}` ...")
            shutil.rmtree(cache_dir)
            print(f"Delete `{model_name}` cache_dir successfully!")
            print(f"Download `{model_name}` successfully!")
            break
        except Exception as e:
            print(f"Error msg: {str(e)}")
            # time.sleep(2)


def main():
    # 或者命令行
    # # 建议将上面这一行写入 ~/.bashrc。若没有写入，则每次下载时都需要先输入该命令
    # export HF_ENDPOINT=https://hf-mirror.com
    # proxychains4 huggingface-cli download --resume-download lamm-mit/x-lora --local-dir x-lora
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", type=str, default="Qwen/Qwen1.5-1.8B-Chat",
        help="model name like: `Qwen/Qwen1.5-1.8B-Chat`")
    parser.add_argument("--save_dir", type=str, default='./', help="model save dir")

    args = parser.parse_args()

    download_huggingface_model(repo_id=args.repo_id, save_dir=args.save_dir)


if __name__ == "__main__":
    main()



