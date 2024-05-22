source /data01/rhino_xsg/software/switch_cuda.sh 12.1
source /data01/rhino_xsg/software/anaconda3/bin/activate py310_cu121_vllm_spec_0.4.2

[ ! -d ./run_logs/controller ] && mkdir -p ./run_logs/controller && echo "controller log created." || echo "controller log dir already existed."
[ ! -d ./run_logs/model_worker ] && mkdir -p ./run_logs/model_worker && echo "model_worker log created." || echo "model_worker log dir already existed."
[ ! -d ./run_logs/api_server ] && mkdir -p ./run_logs/api_server && echo "api_server log created." || echo "api_server log dir already existed."
[ ! -d ./run_logs/gradio ] && mkdir -p ./run_logs/gradio && echo "gradio log created." || echo "gradio log dir already existed."

nohup python controller.py \
  --port 11001  > run_logs/controller/controller_py.log 2>&1 &

# 单卡投机采用加速框架启动脚本
#CUDA_VISIBLE_DEVICES=2 nohup python vllm_worker.py\
#  --model-path "/data01/01model_hub/LLM/Qwen1.5-14B-Chat/" \
#  --trust-remote-code\
#  --worker-address 11002 \
#  --port 11002 \
#  --num-gpus 2 \
#  --dtype "float16" \
#  --gpu-memory-utilization 0.9 \
#  --speculative-model "[ngram]" \
#  --speculative_disable_by_batch_size 32 \
#  --num-speculative-tokens 5 \
#  --speculative-max-model-len 32768 \
#  --ngram-prompt-lookup-max 100 \
#  --ngram-prompt-lookup-min 10 \
#  --use-v2-block-manager \
# --controller-address http://localhost:11001 \
#  --num-gpus 1 > run_logs/model_worker/vllm_worker_py.log 2>&1 &

# 多卡加速
CUDA_VISIBLE_DEVICES=6,7 nohup python vllm_worker.py\
  --model-path "/data01/01model_hub/LLM/Qwen1.5-32B-Chat/" \
  --trust-remote-code\
  --worker-address "http://localhost:11002" \
  --port 11002 \
  --dtype "float16" \
  --gpu-memory-utilization 0.9 \
  --speculative-model "[ngram]" \
  --speculative-disable-by-batch-size 32 \
  --num-speculative-tokens 5 \
  --speculative-max-model-len 32768 \
  --ngram-prompt-lookup-max 100 \
  --ngram-prompt-lookup-min 10 \
  --use-v2-block-manager \
  --controller-address http://localhost:11001 \
  --num-gpus 2 > /dev/null 2>&1 &

  # --num-gpus 2 > run_logs/model_worker/vllm_worker_py.log 2>&1 &

while true; do
  result=$(netstat -tuln | grep ":11002")

  if [ -n "$result" ]; then
    nohup python openai_api_server.py \
    --controller-address "http://localhost:11001" \
    --port 11003 > run_logs/api_server/api_server_py.log 2>&1 &
    nohup python gradio_web_server.py \
    --controller-url "http://localhost:11001" \
    --port 11004 --share > run_logs/gradio/gradio_py.log 2>&1 &
    break
  else:
    sleep 5
  fi
done