#!/usr/bin/env bash
# 6 GPU 노드 전용 실행 스크립트

# 사용할 GPU 지정 (0~5)
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=16   # 예: 프로세스당 8스레드 사용
python3 src/chat.py \
    --max_new_tokens 12288 \
    --temperature 0.0 \
    --top_p 0.75 \
    --repetition_penalty 1.1