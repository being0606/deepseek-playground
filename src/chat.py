import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import SYSTEM_PROMPT, DEMO_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chat with Deepseek-R1-Distill-Llama-8B"
    )
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Hugging Face 모델명 또는 경로")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="한 번에 생성할 최대 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="샘플링 temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus 샘플링 확률 질량")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="반복 패널티")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16","bfloat16","float32"],
                        help="모델 연산에 사용할 dtype")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT,
                        help="대화 시작 시 항상 포함할 시스템 프롬프트")
    return parser.parse_args()


def main():
    args = parse_args()
    system_prompt = args.system_prompt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    torch_dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print(f"챗봇 시작 (exit/quit 입력 시 종료)")

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ("exit","quit"):
            print("챗봇을 종료합니다.")
            break
        user_input = DEMO_PROMPT # 사용자 입력을 대체하는 테스트용 유저인풋 TODO:수정할 예정
        
        
        # 프롬프트
        # prompt = f"<｜User｜>{user_input}<｜Assistant｜><think></think>"
        # Include system prompt if provided
        if system_prompt:
            prompt = f"{system_prompt}\n<｜User｜>{user_input}<｜Assistant｜>"
        else:
            prompt = f"<｜User｜>{user_input}<｜Assistant｜>"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 생성
        start = time.time()
        with torch.no_grad():
            do_sample = args.temperature > 0.0
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
            )
        elapsed = time.time() - start
        gen_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]

        if elapsed > 0 and gen_tokens > 0:
            print(f"[Speed] 토큰당 {elapsed/gen_tokens:.4f}s (총 {gen_tokens}토큰, {elapsed:.2f}s)")
        generated = out[0, inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"Bot : {raw}")


if __name__ == "__main__":
    main()
