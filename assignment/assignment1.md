# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 열혈청강생
- **팀원**: 김종민
- **작성일**: 2026-09-25
- **재현 커맨드**: `python run_mmmu_eval.py --model_path Qwen/Qwen3-VL-4B-Instruct --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --data_root MMMU/MMMU --data_revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 --output_dir ./outputs`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) |
| 추론 백엔드 | Hugging Face Transformers (AutoModelForImageTextToText, PyTorch Native) |
| 사용 GPU | Google Colab 무료 티어 GPU (T4) |
| 실측 peak VRAM | 미측정 (세션 단절 및 재개로 인한 전역 프로파일링 누락) |
| 총 소요 시간 | 900문제 처리를 여러 Colab 세션에 걸쳐 resume 방식으로 진행함 (무료 GPU 한도 소진으로 세션이 중간에 끊겨 재개를 반복함). 마지막 재개 구간(426→900, 474문제) 실측 125.0분. 전체 세션 합산 시간은 정확히 기록되지 않음 — **한계로 8절에 기술** |
| 의존성 | `transformers`, `accelerate`, `qwen-vl-utils[decord]`, `datasets`, `huggingface_hub`, `pandas`. 정확한 버전은 `requirements.txt` 참고 |
| 실행 커맨드 | ```bash\npython run_mmmu_eval.py \\\n  --model_path Qwen/Qwen3-VL-4B-Instruct \\\n  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \\\n  --data_root MMMU/MMMU \\\n  --data_revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \\\n  --output_dir ./outputs\n``` `run_mmmu_eval.py`(본 레포 포함)로 실행. Colab에서 검증한 파이프라인을 argparse 기반 스크립트로 이식함 |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문**:

Multiple-choice 문제:
```
Question: {question}
Options:
{options_block}
Respond with ONLY the single letter of the correct option (e.g. 'A'). Do not provide any explanation or reasoning.
```

Open-ended 문제:
```
Question: {question}
Answer the question directly with a short, precise answer (a number, word, or short phrase). Do not provide any explanation or reasoning.
```

- **출처**: 자체 설계. 초기에는 단순 지시문(`"Answer with the option's letter from the given choices directly."`)을 사용했으나, 파일럿 실험(Accounting 30문제)에서 모델이 설명을 먼저 시작하다 응답이 절단되어 답 letter가 아예 출력되지 않는 현상을 다수 확인함(정답률 13.33%, 오답 26건 중 다수가 "To determine..." 형태로 잘림).
- **선택 이유**: "설명 없이 정답만" 명시적으로 강제하는 지시문으로 교체하여 응답 절단 문제를 해소하기 위함. 동일 조건(30문제)에서 개선 전후 비교 실험으로 효과를 검증함 (2.4절 참고).

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | `true` |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` (패널티 없음) |
| `presence_penalty` | 미지정 — 공식 `generation_config.json`에 해당 키 자체가 없음(`getattr(model.generation_config, "presence_penalty", None)` → `None`). 실질적으로 패널티 미적용 |
| `seed` | `42` (`torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`로 스크립트 시작 시 고정. HF `generate()`/`GenerationConfig`에는 `seed` 필드 자체가 없어 별도 설정 필요) |

- **출처**: `Qwen/Qwen3-VL-4B-Instruct` 리포지토리의 공식 `generation_config.json`. `model.generation_config`를 직접 로드하여 출력, 값을 그대로 채택함(별도 값으로 임의 수정하지 않음).

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `256` |
| 이미지 해상도 처리 | `min_pixels = 256 × 28 × 28 = 200,704`, `max_pixels = 1024 × 28 × 28 = 802,816` |

**선택 근거**:
- `max_new_tokens`: 초기 `32`로 설정했을 때 Accounting 30문제 파일럿에서 정답률 13.33%, 그중 다수가 응답 절단으로 인한 파싱 실패(letter 미출력)로 확인됨. `256`으로 상향 후 동일 표본 재실험 결과 정답률 53.33%로 개선, 응답 절단 0건 확인. 계산 과정이 긴 회계/재무 문제 특성을 고려해 여유를 두고 `256`으로 채택. 다만 이 개선은 프롬프트 강화와 동시에 적용되어, 토큰 증가분만의 독립적인 기여도는 별도로 검증하지 못함 — multiple-choice 응답은 대부분 단일 letter로 종료되어 더 작은 토큰 수로도 충분했을 가능성이 있으나, open-ended 응답 길이 및 `do_sample=True`로 인한 응답 분산에 대한 안전마진으로 256을 유지함.
- 이미지 해상도(`max_pixels`): Colab 무료 GPU(VRAM 14.56 GiB) 환경에서 원본 해상도 그대로 처리 시 특정 고해상도 이미지 문제에서 단일 attention 연산이 27.4 GiB를 요구하며 OOM 발생. 이를 방지하기 위해 `max_pixels`를 제한함. **다만 이 값과 공식 수치(67.4) 사이 정확도 손실의 정량적 관계는 별도로 검증하지 않았으며, 잠재적 trade-off로만 기록함** (7절 참고).

## 4. 채점(파싱) 방식

- **Multiple-choice**: 정규식 기반 3단계 fallback 파서를 자체 구현.
  1. 응답 맨 앞에 오는 `"A"`, `"A."`, `"A)"` 등의 패턴을 우선 탐지
  2. 실패 시 `"answer is A"`, `"answer: A"`, `"option A"` 류 패턴을 응답 전체에서 탐색
  3. 그래도 실패 시 응답 텍스트에 등장하는 첫 letter 후보를 fallback으로 채택
  4. 모두 실패하면 `None` 반환(오답 처리)
- **Open-ended**: 초기에는 텍스트 정규화(소문자화, 공백/마침표 제거) 후 완전일치 방식을 사용했으나, 파일럿 검토 결과 숫자 답의 반올림 차이(예: gold=`2.83`, pred=`2.828`)나 다중 허용 정답이 리스트 문자열로 제공되는 경우(예: gold=`"['$MgS$', 'MgS']"`)를 처리하지 못해 정확도가 저평가됨을 확인. 이에:
  1. gold가 리스트 문자열 형태면 `ast.literal_eval`로 파싱해 여러 허용 후보로 전개
  2. 텍스트 완전일치 우선 확인
  3. 실패 시 응답·정답 모두에서 숫자를 추출(단, 응답에 숫자가 정확히 1개로 명확히 식별될 때만)하여 허용오차(절대오차 0.01 또는 상대오차 2% 중 큰 값) 내 근사 일치 허용
- 위 개선으로 open 문제(53문제) 정답 수가 6개→15개로 증가함을 확인(자체 구현, 별도 출처 없음).

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 53.33 |
| 2 | Agriculture | 30 | 50.00 |
| 3 | Architecture_and_Engineering | 30 | 46.67 |
| 4 | Art | 30 | 66.67 |
| 5 | Art_Theory | 30 | 83.33 |
| 6 | Basic_Medical_Science | 30 | 66.67 |
| 7 | Biology | 30 | 46.67 |
| 8 | Chemistry | 30 | 30.00 |
| 9 | Clinical_Medicine | 30 | 60.00 |
| 10 | Computer_Science | 30 | 53.33 |
| 11 | Design | 30 | 86.67 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 |
| 13 | Economics | 30 | 50.00 |
| 14 | Electronics | 30 | 50.00 |
| 15 | Energy_and_Power | 30 | 40.00 |
| 16 | Finance | 30 | 30.00 |
| 17 | Geography | 30 | 50.00 |
| 18 | History | 30 | 73.33 |
| 19 | Literature | 30 | 76.67 |
| 20 | Manage | 30 | 50.00 |
| 21 | Marketing | 30 | 66.67 |
| 22 | Materials | 30 | 33.33 |
| 23 | Math | 30 | 46.67 |
| 24 | Mechanical_Engineering | 30 | 40.00 |
| 25 | Music | 30 | 26.67 |
| 26 | Pharmacy | 30 | 63.33 |
| 27 | Physics | 30 | 36.67 |
| 28 | Psychology | 30 | 73.33 |
| 29 | Public_Health | 30 | 50.00 |
| 30 | Sociology | 30 | 56.67 |
| | **Overall (macro avg)** | **900** | **53.00** |

계산식: `Overall = mean(30개 과목 accuracy)` (30개 과목 정확도의 단순 평균, 과목별 표본 수는 전부 동일(30개)하므로 micro avg와도 동일함)

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 53.00 |
| 차이 (Δ) | **-14.40** |

## 7. 격차 분석

> 초기 파이프라인(`max_new_tokens=32`, 지시가 약한 프롬프트)에서는 Accounting 표본 정확도가 13.33%에 불과했고, 오답 대부분이 모델이 풀이 과정을 서술하다 응답이 절단되어 정답 letter가 아예 출력되지 않은 경우였다. 프롬프트를 "설명 없이 정답만"으로 강화하고 `max_new_tokens`를 256으로 상향한 결과 동일 표본 정확도가 53.33%로 개선되었으며, 응답 절단 사례는 0건으로 감소했다. 또한 open-ended 문제(전체 53/900) 채점을 exact-match에서 허용오차 기반 근사 비교로 개선해 정답 수가 6→15개로 증가했다. 이 두 개선을 반영한 최종 결과가 macro avg 53.00으로, 여전히 공식 수치(67.4) 대비 14.4점 낮다. 잔여 격차는 (1) 비-greedy 샘플링(`temperature=0.7`, 공식 recipe를 그대로 따른 것이나 공식 벤치마크가 동일 디코딩 조건으로 측정됐는지는 불명), (2) OOM 방지를 위한 `max_pixels` 제한으로 인한 이미지 디테일 손실 가능성(별도 ablation 미실시), (3) 공식 기술 보고서의 평가 환경(표준 프레임워크 사용 여부, 프롬프트 템플릿 등)과 본 실험 조건 간의 세부 차이가 복합적인 원인으로 작용했을 것으로 추정되나, 본 실험에서 개별 검증까지는 수행하지 못했다.
> 
## 8. 기타 특이사항 / 한계 (Optional)

- Colab 무료 GPU 한도 소진으로 세션이 여러 차례 끊겨, 파일 기반 resume(이미 처리된 문제 id를 건너뛰는 방식)으로 재개함. 그 과정에서 로컬(`/content/`) 저장 파일이 세션 종료로 유실된 적이 있어, 이후 Google Drive 마운트로 저장 경로를 변경함.
- 실행 환경(GPU 모델명 상세 스펙, peak VRAM)을 정확히 기록하지 못해 완전한 재현성 확보에는 추가 보완이 필요함.
- 프롬프트 구성 시 이미지 여러 장을 텍스트 앞에 일괄 배치했으며, 원본 질문에 포함된 `<image 1>`, `<image 2>` 등 텍스트 내 위치 지정을 반영하지 않음 — 이미지가 2장 이상인 문제에서 정확도에 영향을 줄 수 있는 단순화.
- `max_pixels` 값이 성능에 미치는 영향은 별도 ablation 실험으로 검증하지 못함. 시간이 더 있었다면 동일 표본에 대해 `max_pixels`를 상향한 재실험으로 원인을 좁혀볼 계획이었음.
- 채점 시 사용한 숫자 근사 비교 로직(허용오차 절대 0.01 / 상대 2%)은 자체 기준이며, MMMU 공식 평가 하네스와 완전히 동일하지 않을 수 있음.
- Colab 노트북에서 검증한 파이프라인을 `run_mmmu_eval.py`(argparse 기반)로 이식함. 노트북과 스크립트 간 로직은 동일하나, 스크립트 버전으로 전체 900문제를 처음부터 재실행하여 결과를 재검증하지는 않음.
