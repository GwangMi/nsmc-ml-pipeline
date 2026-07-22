# NSMC ML Pipeline

PyTorch 프레임워크 내부 이해와 현업 수준 ML 엔지니어링(Git/CI, 데이터 파이프라인,
파인튜닝, 분산학습, RAG, 실험관리, 서빙, 배포)을 하나의 저장소에서 단계별로 쌓아가는 학습 프로젝트.

각 단계는 "도구를 쓰기 전에 원리를 직접 구현해본다"는 원칙을 따른다. 프레임워크가
대신해주는 일을 먼저 손으로 짜봐야, 실제 도구를 쓸 때 뭘 위임하고 있는지 알 수 있다.

전체 시리즈에서 다루는 고정 데이터셋: **NSMC** (Naver Sentiment Movie Corpus, 네이버 영화 리뷰 감성분석)

## 로드맵

| 단계 | 주제 | 원리 체크포인트 | 도구 |
|---|---|---|---|
| 00 | PyTorch 기초 (Tensor/Autograd) | numpy로 MLP backprop 직접 구현 → torch로 재구현 | PyTorch |
| 01 | Git/GitHub 워크플로우 + CI | — | Git, GitHub Actions, pytest, ruff |
| 02 | SQL + Pandas 데이터 파이프라인 | — | SQLite/Postgres, Pandas |
| 03 | 파인튜닝 (LoRA) | attention·LoRA를 numpy/torch로 직접 구현 후 HF+PEFT로 재현 | Transformers, PEFT |
| 04 | 학습 스케일업 | gradient accumulation 수동 구현 후 Accelerate로 비교 | Accelerate, DeepSpeed |
| 05 | RAG | 코사인 유사도 검색 직접 구현 후 벡터DB로 전환 | LangChain/LlamaIndex, FAISS/Chroma |
| 06 | 실험 관리 | — | W&B / MLflow |
| 07 | 추론 서빙 (양자화) | Linear layer 8bit quantize 직접 구현 후 실제 양자화 도구 적용 | vLLM, FastAPI, bitsandbytes |
| 08 | 배포 | — | Docker, Kubernetes, AWS/GCP |

## 진행 상태

- [x] 00 — PyTorch 기초 (`00_fundamentals/`)
- [ ] 01 — Git/GitHub 워크플로우 + CI (지금 이 저장소 세팅 자체가 01단계)
- [ ] 02 — 데이터 파이프라인
- [ ] 03 — 파인튜닝
- [ ] 04 — 학습 스케일업
- [ ] 05 — RAG
- [ ] 06 — 실험 관리
- [ ] 07 — 추론 서빙
- [ ] 08 — 배포

## 개발 환경

모든 단계가 공유하는 가상환경: `C:\Users\<user>\venvs\pytorch_learning` (Windows 경로 길이
제한 때문에 프로젝트 폴더 밖의 짧은 경로에 둔다).

```bash
C:\Users\<user>\venvs\pytorch_learning\Scripts\activate
pip install -r requirements-dev.txt
pytest
ruff check .
```
