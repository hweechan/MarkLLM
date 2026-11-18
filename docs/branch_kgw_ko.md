# BranchKGW: 두 모델을 이용한 경로 선택 워터마킹

작은 draft 모델에서 길이 k의 후보 M개를 샘플링하고, 큰 target 모델의 평균 토큰 로그 확률 Q로 평가합니다. Q >= max(Q) - epsilon인 후보 중 KGW green token 점수 W가 가장 높은 후보를 확정합니다. W가 같으면 Q가 높은 후보, 그것도 같으면 먼저 생성된 후보를 선택합니다. 다음 블록은 확정된 문맥에서 시작합니다.

- Q = mean(log p_target(candidate_token | preceding_context)); 원래 target softmax 기준이며 draft temperature/top-p는 적용하지 않습니다.
- W = sum(green_indicator - gamma); EOS로 조기 종료된 후보는 실제 길이만 사용합니다. 길이가 다른 후보의 점수 분산도 다르므로 EOS 빈도와 출력 길이를 함께 분석해야 합니다.
- 기본값: M=4, k=4, epsilon=0.2 nats/token, gamma=0.5. epsilon은 후보 중 최고 점수와의 상대 조건이며 절대 품질 보장이 아닙니다.
- `quality-only`는 같은 후보 생성 방식에서 Q가 가장 높은 후보를 선택하는 대조군입니다. 일반적인 target 단독 생성과 다릅니다.
- 표준 speculative sampling이 아니며 target 분포 보존이나 가속을 보장하지 않습니다. 초기 구현은 후보를 순차 평가하고 블록 간 KV cache를 재사용하지 않습니다.
- 같은 token ID와 special token 체계의 decoder-only 모델만 지원합니다. 서로 다른 GPU에 배치할 수 있지만 각 모델은 단일 장치에 로드합니다. AutoWatermark 등록 대신 `watermark.branch_kgw.BranchKGW`를 직접 사용합니다.

## 1. 새 SSH 서버 설치

Linux, Python 3.10 이상(권장 3.11/3.12), Git, NVIDIA GPU를 가정합니다. 아래 명령은 서버에 SSH 접속한 뒤 실행합니다.

```bash
nvidia-smi
git clone https://github.com/hweechan/MarkLLM.git
cd MarkLLM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 12.1 wheel을 지원하는 NVIDIA 드라이버 환경의 예시입니다.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-branch.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -p 'test_branch_kgw.py' -v
```

`venv` 모듈이 없다면 서버 관리자에게 Python venv 패키지 설치를 요청하세요. CUDA wheel은 서버 드라이버에 맞게 선택해야 합니다. CPU 전용 검증은 PyTorch 설치 URL을 `https://download.pytorch.org/whl/cpu`로 바꾸면 됩니다. 이 기능에는 전체 `requirements.txt`나 vLLM 설치가 필요하지 않습니다. 테스트는 모델 다운로드 없이 작은 무작위 GPT-2를 CPU에서 실행합니다.

첫 실행은 Hugging Face에서 두 모델을 다운로드하므로 인터넷과 디스크 공간이 필요합니다. 필요하면 `export HF_HOME=/path/to/your/cache`로 캐시 경로를 지정하세요. 모델 이름 대신 다운로드한 로컬 모델 디렉터리를 전달할 수도 있습니다.

## 2. 첫 실행: 단일 GPU

```bash
python -m examples.branch_kgw \
  --target Qwen/Qwen2.5-1.5B \
  --draft Qwen/Qwen2.5-0.5B \
  --target-device cuda:0 --draft-device cuda:0 \
  --dtype bfloat16 \
  --prompt 'Explain why the sky appears blue in clear, simple terms.' \
  --candidates 4 --block-size 4 --epsilon 0.2 \
  --max-new-tokens 128 --seed 42 \
  --output outputs/watermarked.json
```

이는 작은 규모의 시작용 모델 조합이며 성능을 검증한 조합은 아닙니다. 두 모델의 BF16 가중치만 대략 4GB이고 실행에는 logits, KV cache 등의 추가 메모리가 필요합니다. 여유 있는 12~16GB급 GPU를 시작점으로 고려하되 실제 요구량은 문맥 길이와 후보 수 등에 따라 달라집니다. BF16 미지원 GPU는 `--dtype float16`을 사용하세요. CPU는 두 장치를 모두 `cpu`, dtype을 `float32`로 바꿉니다.

더 큰 target을 쓰려면 `--target Qwen/Qwen2.5-7B`로 변경할 수 있습니다. 같은 모델 계열이어도 tokenizer 호환성은 실제 로드 시 확인합니다. OOM이면 먼저 후보 수와 문맥/출력 길이를 줄이거나 작은 target을 사용하세요.

## 3. 두 GPU에 분리

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m examples.branch_kgw \
  --target Qwen/Qwen2.5-7B --draft Qwen/Qwen2.5-0.5B \
  --target-device cuda:0 --draft-device cuda:1 \
  --max-new-tokens 256 --output outputs/two_gpu.json
```

각 모델이 해당 GPU 메모리에 들어가야 합니다. 두 GPU로 나누어도 현재 구현은 평가와 초안 생성을 겹쳐 실행하지 않습니다.

SSH 연결이 끊겨도 실행하려면 `tmux new -s branch-wm` 안에서 가상환경 활성화 후 명령을 실행하고 `Ctrl-b`, `d`로 분리하세요. 재접속 후 `tmux attach -t branch-wm`으로 돌아옵니다.

## 4. 대조군과 파라미터

```bash
python -m examples.branch_kgw \
  --mode quality-only --seed 42 \
  --candidates 4 --block-size 4 --epsilon 0.2 \
  --max-new-tokens 128 --output outputs/quality_only.json
```

비교 실행의 모델, prompt, seed, temperature, top-p, 후보 수, 블록 길이를 맞추세요. 같은 seed라도 선택된 문맥이 달라진 이후 후보는 달라집니다. 동일 seed가 끝까지 동일 후보를 의미하지 않습니다.

- `--epsilon 0`: 최고 Q 후보만 허용(동점 후보에서는 WM 선택 가능).
- epsilon을 키우면 WM 선택의 자유가 커지지만 후보 내 품질 손실 허용 폭도 커집니다.
- `--candidates 1`: 경로 선택을 통한 WM 효과가 없는 sanity check.
- `--block-size 1`: 토큰 단위 후보 선택 비교.
- `--hash-key`: KGW 키. 기본 키는 공개된 실험용 값입니다. 결과 JSON에도 재현을 위해 키가 기록되므로 비밀 키 실험 결과는 공개하지 마세요.
- 여러 prompt와 seed에서 고정 오탐률의 검출력, 텍스트 품질, 생성 길이, 처리량을 비교하세요. 한 문장의 z-score로 효과를 판단하지 마세요.

## 5. 결과와 검출

JSON에는 continuation, 정확한 prompt/generated token IDs, 후보별 Q/W, 품질 조건을 통과한 후보, 선택 index, 설정, 모델 revision, 실행 시간, GPU 최대 allocated 메모리가 저장됩니다. 처리 시간에는 생성과 검출/trace 작성이 포함되고 모델 로딩은 제외됩니다. 정밀한 속도 비교는 warm-up과 반복 측정이 추가로 필요합니다.

`detection_with_prompt`는 원래 prompt를 문맥으로만 사용하며 prompt 토큰은 점수에 포함하지 않습니다. `detection_without_prompt`는 생성 토큰만으로 계산하며 첫 prefix_length 토큰을 건너뜁니다. 점수에 사용할 토큰이 없으면 score는 null입니다. EOS 등 실제 생성된 special token도 token-ID 점수에는 포함됩니다.

이 레포의 KGW greenlist는 PyTorch 장치별 난수 생성기를 사용하므로 생성과 검출의 장치 종류(CPU/CUDA)를 맞추세요. CPU와 CUDA 사이의 greenlist 일치를 가정하면 안 됩니다. 재현 시 PyTorch 버전도 유지하세요.

텍스트만으로는 상속된 `detect_watermark(text)`를 사용할 수 있지만, skip_special_tokens 및 재토큰화로 정확한 생성 ID와 달라질 수 있습니다. 재현 실험에서는 JSON의 token IDs와 `detect_token_ids()`를 권장합니다. 전체 prompt+출력을 그대로 검출하면 prompt까지 계산되므로 주의하세요.

기존 KGW의 z_threshold=4는 임시 기준입니다. `is_watermarked`는 이 임시 기준의 결과이고 `threshold_is_calibrated=false`로 표시합니다. 경로 선택과 반복 문맥은 통계적 의존성을 만들 수 있습니다. 별도 quality-only calibration 데이터로 원하는 FPR의 임계값을 정한 후, 분리된 평가 데이터에서 TPR/FPR을 측정해야 합니다. 이 구현에는 자동 calibration이나 품질 보장 fallback이 포함되지 않습니다.

## 6. Python API

```python
from utils.transformers_config import TransformersConfig
from watermark.branch_kgw import BranchKGW, BranchOptions

# target, draft, tokenizer, draft_tokenizer는 미리 로드한 HF 객체
config = TransformersConfig(target, tokenizer, device='cuda:0')
wm = BranchKGW('config/KGW.json', config, draft, draft_tokenizer,
               BranchOptions(candidates=4, block_size=4, epsilon=0.2))
result = wm.generate('Explain the greenhouse effect.', watermarked=True)
control = wm.generate('Explain the greenhouse effect.', watermarked=False)
print(result['continuation'])
```

KGWConfig의 delta는 이 경로 선택 알고리즘에서 사용하지 않습니다. draft logits에 KGW bias를 넣지 않으며 greenlist/검출 규칙만 재사용합니다. BranchOptions가 branch 생성 설정을 담당하고 TransformersConfig.gen_kwargs는 사용하지 않습니다.

## 7. 업데이트

```bash
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -r requirements-branch.txt
python -m unittest discover -s tests -p 'test_branch_kgw.py' -v
```

현재 목표는 방법론의 기준 구현입니다. 이후 실제 모델 실험으로 강도를 확인한 다음 후보 배치 평가, KV cache 재사용, 절대 품질 fallback, 검출 calibration을 확장할 수 있습니다.
