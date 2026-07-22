# Stage 1 — 기초 (Tensor & Autograd)

목표: PyTorch의 핵심 개념(Tensor, autograd, nn.Module, optimizer)을 이해하고,
"직접 구현한 backprop"과 "PyTorch가 자동으로 해주는 것"의 차이를 몸으로 느낀다.

## 진행 순서

1. `numpy_mlp/` — numpy만으로 2-layer MLP를 만들고 forward/backward를 직접 미분해서 구현.
   Fashion-MNIST를 분류한다. autograd 없이 gradient를 손으로 유도해봐야
   PyTorch의 `loss.backward()`가 뭘 대신해주는지 감이 온다.
2. `torch_mlp/` — 같은 모델을 `torch.nn.Module` + `autograd` + `optim.SGD`로 재구현.
   numpy 버전과 결과(정확도)가 비슷하게 나오는지 비교한다.

## 환경 설정

가상환경은 `pytorch_learning` 시리즈 전체가 공유하도록 짧은 경로(`C:\Users\<user>\venvs\pytorch_learning`)에 만들어져 있다.
프로젝트 폴더 안(`stage1_fundamentals\.venv`)에 만들면 torch 패키지 내부의 깊은 라이선스 파일 경로가
Windows의 260자 경로 길이 제한을 넘어 설치가 실패하기 때문이다.

```bash
# 최초 1회 (이미 생성되어 있으면 생략)
python -m venv C:\Users\<user>\venvs\pytorch_learning
C:\Users\<user>\venvs\pytorch_learning\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
C:\Users\<user>\venvs\pytorch_learning\Scripts\activate

# numpy 버전 (외부 데이터 다운로드 없이 torchvision으로 데이터만 불러옴)
python numpy_mlp/train.py

# PyTorch 버전
python torch_mlp/train.py
```

## 체크포인트 (이 단계에서 스스로 답할 수 있어야 하는 질문)

- forward pass에서 저장해둔 값들이 backward pass에서 왜 필요한가?
- `loss.backward()` 호출 시 어떤 그래프를 타고 gradient가 흐르는가?
- learning rate를 10배 올리면 numpy 버전과 torch 버전 둘 다 왜 발산하는가?
- `optimizer.zero_grad()`를 빼먹으면 어떤 일이 일어나는가? (직접 지워보고 확인)
