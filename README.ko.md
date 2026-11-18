# 학생 설문 기반 우울 라벨 분류

[English](README.md) · [사례 연구](PORTFOLIO.ko.md)

502명이 응답한 학생 설문에서 어떤 생활·스트레스 응답이 우울 라벨(Yes/No)과 함께 나타나는지 로지스틱 회귀로 탐색한 분석입니다. 인천대학교 프로그래밍언어 과목 기말 프로젝트(2024년 11~12월)입니다.

> 이 프로젝트는 분석 탐색용입니다. 진단 도구가 아니며, 임상 판단이나 자동화된 고위험 의사결정에 사용하면 안 됩니다.

**왜 중요한가:** 정신건강 설문 데이터는 작고 민감하므로, 불투명한 모델로 점수를 높이기보다 신호를 담은 소수의 응답을 찾고 그 계수를 누구나 확인할 수 있게 하는 것이 목표입니다.

**결과:** 설문 변수 4개를 쓴 로지스틱 회귀가 101건 holdout(seed 고정 80/20 분할)에서 **accuracy 0.9010**, **weighted F1 0.90**을 기록했습니다. 같은 holdout에서 항상 다수 라벨을 예측하면 **0.5248**입니다. 하나의 설문 표본에서 한 번 나눈 결과이며 임상 성능이 아닙니다.

## 데이터

- `data/Depression Student Dataset.csv`: 502행, 11개 컬럼, 결측 없음. 라벨은 균형에 가깝습니다(Yes 252, No 250). 출처: 분석 노트북에 기재된 [Kaggle Depression Student Dataset](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset/data).
- 변수: 성별, 나이, 학업 압박, 학업 만족도, 수면 시간, 식습관, 자살사고 이력, 공부 시간, 재정 스트레스, 정신질환 가족력, 우울 라벨.

## 접근 방법

```mermaid
flowchart LR
    A[학생 설문 CSV\n502행] --> B[타깃·순서형·범주형 인코딩]
    B --> C[Chi-square·Spearman 탐색]
    C --> D[선택 변수 4개]
    D --> E[80/20 로지스틱 회귀 분할\nseed 42]
    E --> F[Accuracy·분류 리포트\n혼동행렬·p-value]
```

- `src/preprocessing.py`는 `Depression` 타깃, 순서형 3개, 범주형 5개를 인코딩합니다.
- `src/train.py`는 식습관, 자살사고 이력, 학업 압박, 재정 스트레스를 사용해 지표용 scikit-learn 로지스틱 회귀와 계수·p-value용 statsmodels Logit을 함께 실행합니다.
- `src/generate_plots.py`는 Spearman, countplot, learning curve 그림을 만듭니다.

모든 예측을 계수 4개로 추적할 수 있도록 더 유연한 모델 대신 로지스틱 회귀를 택했고, 다른 모델과는 비교하지 않았습니다. 노트북은 라벨과의 chi-square 검정에서 p<0.05인 범주형 변수(식습관, 자살사고 이력)와 Spearman |ρ|가 0.3 이상인 순서형 변수(학업 압박 0.48, 재정 스트레스 0.30)를 남겼습니다.

## 결과

| 모델 | 변수 | Accuracy | Weighted F1 | 평가 조건 |
|---|---|---:|---:|---|
| 다수 라벨("Yes") | 없음 | 0.5248 | - | 같은 101건 holdout(양성 53건) |
| 로지스틱 회귀 | 선택 변수 4개 | 0.9010 | 0.90 | seed 고정 80/20 분할, 비층화 |

혼동행렬은 `[[42, 6], [4, 49]]`로, 양성 53건 중 49건과 음성 48건 중 42건을 맞혔습니다. Logit 계수 4개는 학습 요약에서 모두 p<0.001입니다. 유지 중인 스크립트를 포함된 데이터로 2026-08-20에 다시 실행했으며, 전체 출력은 [`results/model_summary.txt`](results/model_summary.txt)에 있습니다. 기준선 행은 이 파일의 클래스별 support에서 계산했습니다.

## 결과가 말해 주는 것과 아닌 것

- 무작위 분할 한 번이며 반복 검증이나 외부 검증이 아닙니다. 신뢰구간, 보정, 하위집단 분석도 없습니다. 노트북의 학습 정확도는 0.8354로 holdout 0.9010보다 낮아, 분할이 유리하게 나왔을 가능성이 있습니다.
- 변수 4개는 분할 전에 502행 전체에 대한 검정으로 골랐으므로 holdout이 변수 선택과 완전히 독립적이지 않고, 정확도가 낙관적일 수 있습니다.
- 자살사고 이력은 라벨이 측정하려는 개념과 겹치며 계수가 가장 큽니다. 이 변수에 정확도가 얼마나 의존하는지는 확인하지 않았습니다.
- 한 설문 표본 안의 연관성이며 원인이나 임상적 위험이 아닙니다.

## 보관된 시각 자료

![인코딩된 우울 라벨과 탐색 변수의 상관](results/figures/correlation_with_depression.png)

*그림 1. 포함된 설문 데이터의 탐색적 상관입니다. 피처 검토에는 사용했지만 임상·인과 관계를 뜻하지는 않습니다.*

![자살사고 이력 응답별 우울 라벨 분포](results/figures/countplot_suicidal_thoughts.png)

*그림 2. 한 설문 응답에 따른 라벨 분포를 서술적으로 보여 줍니다. 진단 규칙이 아닙니다.*

![유지 중인 로지스틱 회귀 흐름의 learning curve](results/figures/learning_curve.png)

*그림 3. 학습 데이터 크기별 학습 정확도와 5-fold 교차 검증 정확도입니다. 작은 데이터에서의 안정성을 보는 그림이며, 위의 101건 holdout 결과와는 별개입니다.*

## 실행

Python 3.13.2에서 스크립트 컴파일을 확인했습니다. 이 README를 위해 다시 실행하지 않았으므로 `results/model_summary.txt`는 2026-08-20 출력 그대로입니다.

```powershell
pip install -r requirements.txt
python src\preprocessing.py
python src\train.py
python src\generate_plots.py
```

`src/train.py`는 지표를 출력하고 `results/model_summary.txt`를 쓰며, `src/generate_plots.py`는 그림 5개를 `results/figures/`에 저장합니다. `src/config.py`가 데이터 경로, 출력 디렉터리, 분할 비율, seed, 선택 변수를 관리합니다.

## 저장소 구조

```text
src/                  전처리, 학습, 시각화, 공통 설정
data/                 설문 CSV
results/              유지 중인 스크립트의 모델 요약과 그림
notebooks/            출력이 저장된 원본 탐색 노트북
doc/                  프로젝트 보고서(한국어 PDF)
images/               results/figures/와 같은 그림 사본
```

## 문서

- [포트폴리오 사례 연구](PORTFOLIO.ko.md)
- [프로젝트 리뷰](docs/PROJECT_REVIEW.md)
- [아키텍처](docs/ARCHITECTURE.md)
