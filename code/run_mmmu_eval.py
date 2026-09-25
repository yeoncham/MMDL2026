#!/usr/bin/env python3
"""
MMMU-val Baseline Evaluation — Qwen3-VL-4B-Instruct

Colab 노트북에서 검증된 파이프라인(프롬프트, sampling recipe, 채점 로직)을
그대로 옮긴 커맨드라인 스크립트입니다.

사용 예:
    python run_mmmu_eval.py \
        --model_path Qwen/Qwen3-VL-4B-Instruct \
        --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
        --data_root MMMU/MMMU \
        --data_revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \
        --output_dir ./outputs

세션이 끊겨도 --output_dir/raw_logs.jsonl에 이미 처리된 문제 id가 남아있으면
같은 커맨드를 다시 실행하는 것만으로 자동으로 이어서 진행됩니다(resume).
"""

import argparse
import ast
import gc
import json
import os
import re
import string
import time
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]


# --------------------------------------------------------------------------
# CLI 인자
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MMMU-val baseline evaluation")
    p.add_argument("--model_path", type=str, required=True,
                    help="예: Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", type=str, required=True,
                    help="모델 리포지토리 revision(커밋 해시)")
    p.add_argument("--data_root", type=str, required=True,
                    help="예: MMMU/MMMU")
    p.add_argument("--data_revision", type=str, required=True,
                    help="데이터셋 리포지토리 revision(커밋 해시)")
    p.add_argument("--output_dir", type=str, required=True,
                    help="raw_logs.jsonl과 결과표가 저장될 디렉토리")

    # 생성 설정 (기본값은 리포트에 기록된 값과 동일)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--subjects", type=str, nargs="+", default=None,
                    help="특정 과목만 실행하고 싶을 때 지정 (기본: 30개 전부)")
    p.add_argument("--no_resume", action="store_true",
                    help="지정 시 기존 raw_logs.jsonl을 무시하고 처음부터 다시 실행")

    return p.parse_args()


# --------------------------------------------------------------------------
# 프롬프트 구성 (노트북 셀 5)
# --------------------------------------------------------------------------
def build_prompt_and_images(example):
    question = example["question"]
    q_type = example["question_type"]  # "multiple-choice" or "open"

    images = []
    for i in range(1, 8):
        img = example.get(f"image_{i}")
        if img is not None:
            images.append(img.convert("RGB") if isinstance(img, Image.Image) else img)

    if q_type == "multiple-choice":
        raw_options = example["options"]
        if isinstance(raw_options, str):
            raw_options = eval(raw_options)  # MMMU 데이터셋 특성상 str(list) 형태
        letters = list(string.ascii_uppercase[: len(raw_options)])
        options_block = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, raw_options))
        prompt_text = (
            f"Question: {question}\n"
            f"Options:\n{options_block}\n"
            f"Respond with ONLY the single letter of the correct option (e.g. 'A'). "
            f"Do not provide any explanation or reasoning."
        )
    else:  # open
        letters = []
        prompt_text = (
            f"Question: {question}\n"
            f"Answer the question directly with a short, precise answer "
            f"(a number, word, or short phrase). Do not provide any explanation or reasoning."
        )

    return prompt_text, images, letters, q_type


# --------------------------------------------------------------------------
# 생성 함수 (노트북 셀 6)
# --------------------------------------------------------------------------
def run_one(example, model, processor, gen_kwargs):
    prompt_text, images, letters, q_type = build_prompt_and_images(example)
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text, letters, q_type


# --------------------------------------------------------------------------
# Multiple-choice 파싱 (노트북 셀 7)
# --------------------------------------------------------------------------
def parse_mc_answer(generated_text, letters):
    text = generated_text.strip()

    m = re.match(r"^\(?([A-J])\)?[\.\:\)]?", text)
    if m and m.group(1) in letters:
        return m.group(1)

    m = re.search(r"(?:answer is|answer:|option)\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()

    for ch in text:
        if ch in letters:
            return ch
    return None


def parse_answer(generated_text, letters, q_type):
    if q_type == "multiple-choice":
        return parse_mc_answer(generated_text, letters)
    return generated_text.strip()  # open은 raw 저장 후 채점 단계에서 별도 처리


# --------------------------------------------------------------------------
# Open-ended 채점 (노트북 셀 9)
# --------------------------------------------------------------------------
def gold_candidates(gold_raw):
    s = str(gold_raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(x) for x in ast.literal_eval(s)]
        except Exception:
            return [s]
    return [s]


def parse_number(s):
    s = str(s).strip()
    m = re.fullmatch(r"-?\d+\.?\d*\s*/\s*\d+\.?\d*\s*\w*", s)
    if m:
        num_str = re.match(r"-?\d+\.?\d*", s).group()
        denom_str = re.search(r"/\s*(\d+\.?\d*)", s).group(1)
        try:
            return float(num_str) / float(denom_str)
        except Exception:
            return None
    nums = re.findall(r"-?\d+\.?\d*", s)
    if len(nums) == 1:
        try:
            return float(nums[0])
        except Exception:
            return None
    return None


def is_open_correct(pred, gold_raw, rel_tol=0.02, abs_tol=0.01):
    if pred is None:
        return False
    candidates = gold_candidates(gold_raw)

    pred_norm = pred.strip().lower().rstrip(".")
    for c in candidates:
        if pred_norm == c.strip().lower().rstrip("."):
            return True

    pred_num = parse_number(pred)
    for c in candidates:
        gold_num = parse_number(c)
        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) <= max(abs_tol, abs(gold_num) * rel_tol):
                return True
    return False


def is_correct(pred, gold_raw, q_type):
    if q_type == "multiple-choice":
        return pred == gold_raw
    return is_open_correct(pred, gold_raw)


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "raw_logs.jsonl")

    subjects = args.subjects if args.subjects else SUBJECTS

    # ---- 모델/프로세서 로드 ----
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[load] model={args.model_path}@{args.model_revision}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print(model.generation_config)
    print(f"[dtype] {model.dtype}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # ---- 데이터셋 로드 ----
    from datasets import load_dataset

    print(f"[load] dataset={args.data_root}@{args.data_revision}, subjects={len(subjects)}")
    all_splits = []
    for subj in subjects:
        ds = load_dataset(args.data_root, subj, split="validation", revision=args.data_revision)
        ds = ds.add_column("subject", [subj] * len(ds))
        all_splits.append(ds)
        print(f"  {subj}: {len(ds)}")

    if subjects == SUBJECTS:
        total = sum(len(d) for d in all_splits)
        assert total == 900, f"과목당 30문제, 총 900문제여야 함 (현재 {total})"

    # ---- resume: 이미 처리된 id 불러오기 ----
    done_ids = set()
    if not args.no_resume and os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"[resume] 이미 처리된 문제 수: {len(done_ids)} -> 이어서 진행")
    elif args.no_resume and os.path.exists(log_path):
        os.remove(log_path)
        print("[resume] --no_resume 지정됨: 기존 로그 삭제 후 처음부터 실행")

    # ---- 추론 루프 ----
    start = time.time()
    with open(log_path, "a") as f:
        for ds in all_splits:
            subj_name = ds[0]["subject"]
            for ex in tqdm(ds, desc=subj_name):
                if ex.get("id") in done_ids:
                    continue
                try:
                    gen_text, letters, q_type = run_one(ex, model, processor, gen_kwargs)
                    pred = parse_answer(gen_text, letters, q_type)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    gen_text, pred, q_type = "[OOM_ERROR]", None, ex["question_type"]

                row = {
                    "id": ex.get("id"),
                    "subject": ex["subject"],
                    "q_type": q_type,
                    "gold": ex["answer"],
                    "pred": pred,
                    "raw_output": gen_text,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - start
    print(f"[done] elapsed: {elapsed/60:.1f} min")

    # ---- 최종 채점 및 결과표 ----
    with open(log_path) as f:
        all_rows = [json.loads(l) for l in f]

    results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_rows:
        subj = r["subject"]
        correct = is_correct(r["pred"], r["gold"], r["q_type"])
        results[subj]["total"] += 1
        results[subj]["correct"] += int(correct)

    rows_out = []
    for subj in subjects:
        c, t = results[subj]["correct"], results[subj]["total"]
        acc = round(c / t * 100, 2) if t else 0.0
        rows_out.append({"Subject": subj, "Data Num": t, "Acc": acc})

    df = pd.DataFrame(rows_out)
    overall = df["Acc"].mean()

    print(df.to_string(index=False))
    print(f"\nOverall (macro avg): {overall:.2f}")

    df.to_csv(os.path.join(args.output_dir, "results_by_subject.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"overall_macro_avg": overall, "elapsed_min": elapsed / 60}, f, indent=2)


if __name__ == "__main__":
    main()
