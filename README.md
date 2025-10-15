# medicalCostAI

나이, BMI, 성별, 자녀 수, 흡연 여부를 고려한 연간 의료비용 예측 인공지능 프로젝트입니다.

![의료비용 예측 결과](https://github.com/WhiteHerb/medicalCostAI/assets/69900670/6c59b6f2-a1fb-4171-94fd-74e14cbd9ef9)

## 개요

이 프로젝트는 TensorFlow와 Keras를 사용하여 개인의 특성을 기반으로 연간 의료비용을 예측하는 머신러닝 모델입니다. 학교 자율과제의 일부로 제작되었으며, 회귀(Regression) 문제를 딥러닝으로 해결합니다.

## 주요 기능

- **다층 신경망 모델**: 7개 층으로 구성된 Dense 네트워크
- **정규화 기법**: L2 regularization과 Dropout을 통한 과적합 방지
- **실시간 예측**: 사용자 입력을 받아 즉시 의료비용 예측
- **모델 저장/로드**: 학습된 모델 재사용 가능
- **TensorBoard 지원**: 학습 과정 시각화

## 입력 특성 (Features)

| 특성 | 설명 | 데이터 타입 |
|------|------|------------|
| age | 나이 | Integer |
| bmi | 체질량지수 (Body Mass Index) | Float |
| children | 자녀 수 | Integer |
| sex | 성별 (male/female) | One-hot encoded |
| smoker | 흡연 여부 (yes/no) | One-hot encoded |

**목표 변수**: `charges` (연간 의료비용, USD)

## 기술 스택

### 머신러닝/딥러닝
- **TensorFlow** - 딥러닝 프레임워크
- **Keras** - 고수준 신경망 API
- **scikit-learn** - R² score 계산

### 데이터 처리
- **Pandas** - 데이터 분석 및 전처리
- **NumPy** - 수치 연산

### 시각화
- **TensorBoard** - 학습 과정 모니터링
- **Colab** - 클라우드 기반 Jupyter 환경

## 모델 아키텍처

```python
Sequential Model:
- Dense(7, activation='relu', input_dim=7)          # 입력층
- Dense(10, activation='relu', L2 regularization)    # 은닉층 1
- Dropout(0.1)                                       # 드롭아웃 1
- Dense(20, activation='relu', L2 regularization)    # 은닉층 2
- Dropout(0.05)                                      # 드롭아웃 2
- Dense(20, activation='relu', L2 regularization)    # 은닉층 3
- Dropout(0.1)                                       # 드롭아웃 3
- Dense(10, activation='relu')                       # 은닉층 4
- Dense(5, activation='relu')                        # 은닉층 5
- Dense(1)                                           # 출력층 (비용 예측)
```

**총 파라미터**: 1,047개

### 하이퍼파라미터

- **Optimizer**: Adam
- **Loss Function**: Mean Absolute Error (MAE)
- **Batch Size**: 25
- **Epochs**: 200
- **Validation Split**: 10%
- **L2 Regularization**: 0.001
- **Dropout Rates**: 0.05 ~ 0.1

## 데이터셋

### 출처
`insurance.csv` - 의료 보험 데이터셋
- **총 샘플 수**: 1,338개
- **특성 수**: 8개 (전처리 후 7개)

### 전처리 과정

1. **불필요한 컬럼 제거**: `region` 컬럼 삭제
2. **원-핫 인코딩**: 범주형 변수 (sex, smoker) 변환
3. **타겟 분리**: `charges` 컬럼을 목표 변수로 분리
4. **데이터셋 생성**: `tf.data.Dataset`으로 변환
5. **셔플링 및 배치**: 학습 효율성 향상

전처리 후 데이터 형태:
```
age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes
19, 27.9, 0, 1, 0, 0, 1  → charges: 16884.924
```

## 설치 및 실행

### Google Colab (권장)

1. [GitHub에서 노트북 열기](https://github.com/herbpot/medicalCostAI)
2. "Open in Colab" 배지 클릭
3. 셀을 순서대로 실행

### 로컬 환경

**필수 요구사항**:
- Python 3.7 이상
- TensorFlow 2.x

**설치**:
```bash
# 저장소 클론
git clone https://github.com/herbpot/medicalCostAI.git
cd medicalCostAI

# 의존성 설치
pip install tensorflow pandas numpy scikit-learn pydot
```

**Jupyter 노트북 실행**:
```bash
jupyter notebook medical_cost_AI.ipynb
```

## 사용 방법

### 1. 모듈 불러오기 및 데이터 준비
```python
import tensorflow as tf
from tensorflow.python import keras
import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv('insurance.csv')
del data['region']
data = pd.get_dummies(data)
target = data.pop('charges')
```

### 2. 모델 학습
```python
model = create_model()
model.fit(
    x=data.to_numpy(),
    y=target,
    batch_size=25,
    epochs=200,
    validation_split=0.1
)
```

### 3. 예측하기

**코드로 예측**:
```python
# 18세, BMI 23, 자녀 0명, 남성, 비흡연
x = np.array([18, 23, 0, 0, 1, 1, 0]).reshape(-1, 7)
prediction = model.predict(x)[0][0]
cost = (prediction / 16) * 1100  # 원화 환산
print(f"예상 연간 의료비: {cost:.0f}원")
```

**대화형 입력**:
```python
age = input('나이를 입력하세요 >>>')
bmi = input('BMI를 입력하세요 >>>')
children = input('자녀 수를 입력하세요 >>>')
sex = input('성별을 입력하세요 (남/여) >>>')
smoker = input('흡연 여부를 입력하세요 (y/n) >>>')
```

### 4. 모델 저장 및 로드

**저장**:
```python
model.save('./models/MedicalCost')
```

**로드**:
```python
model = keras.models.load_model('./models/MedicalCost')
```

## 성능 평가

### R² Score
모델의 설명력을 나타내는 R² score를 사용:

```python
from sklearn.metrics import r2_score

# 테스트 데이터로 평가
r2 = r2_score(test_y, predictions)
print(f"R² Score: {r2:.4f}")
```

### Loss Metric
- **Mean Absolute Error (MAE)**: 예측값과 실제값의 평균 절대 오차

## TensorBoard 시각화

학습 과정을 실시간으로 모니터링:

```python
%load_ext tensorboard
%tensorboard --logdir=./training_log
```

**확인 가능한 정보**:
- Training Loss/MAE
- Validation Loss/MAE
- Epoch별 성능 변화

## 과적합 방지 기법

1. **L2 Regularization**: 가중치 크기에 페널티 부여 (λ=0.001)
2. **Dropout**: 학습 시 일부 노드 무작위로 비활성화 (5%~10%)
3. **Validation Split**: 검증 데이터로 일반화 성능 모니터링 (10%)
4. **Early Stopping**: (선택적) 검증 손실 증가 시 학습 중단

## 프로젝트 구조

```
medicalCostAI/
├── medical_cost_AI.ipynb    # 메인 노트북
├── insurance.csv             # 데이터셋 (별도 다운로드 필요)
├── models/
│   └── MedicalCost/          # 학습된 모델 저장 위치
├── training_log/             # TensorBoard 로그
├── LICENSE
└── README.md
```

## 결과 분석

- 나이가 많을수록 의료비용 증가 경향
- BMI가 높을수록 의료비용 증가
- 흡연자의 의료비용이 비흡연자보다 현저히 높음
- 자녀 수는 상대적으로 영향이 적음

## 개선 가능한 부분

- [ ] 더 많은 데이터셋으로 학습
- [ ] Feature Engineering (파생 변수 생성)
- [ ] 앙상블 모델 적용 (XGBoost, Random Forest)
- [ ] 하이퍼파라미터 튜닝 (Keras Tuner)
- [ ] Cross-Validation 적용
- [ ] 다른 손실 함수 실험 (RMSE, Huber Loss)

## 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras API 문서](https://keras.io/)
- [Insurance Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여

학습 프로젝트이지만 이슈 및 개선 제안은 환영합니다!

## 작성자

국민대학교 소프트웨어학부 학생
