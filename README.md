# LightGBM 기반 카드 이상거래 탐지

`카드이상거래탐지_프로젝트계획서.docx`의 설계를 그대로 구현한 코드와 결과입니다.

## 실행 방법

```powershell
pip install -r requirements.txt
python -m src.build_data      # 1~2주차: 점검·시간순 분할·특징 추출 (약 1분)
python -m src.experiments     # 3~6주차: 가중치·보정, 실험 A~D, 임계값 (약 3분)
streamlit run app.py          # 7. 대시보드
```

원본 데이터는 Kaggle `kartik2112/fraud-detection`(Sparkov)의 `fraudTrain.csv`, `fraudTest.csv`를 `data/raw/`에 둡니다.
`data/` 폴더와 `outputs/test_scores.parquet`는 용량·재배포 조건 때문에 저장소에 포함하지 않으며, 위 명령으로 다시 만들 수 있습니다.

## 확장판 (v2, 진행 중) — 설계: [docs/확장설계서.md](docs/확장설계서.md)

```powershell
python -m src.sources.sparkov      # 미국 데이터를 공통 스키마로 변환 (build_data 이후)
python -m src.sources.aihub_card   # 한국 데이터 변환 (원본이 없으면 안내만 출력)
python -m src.build_features_v2    # 신규 특징 생성 + 무작위 2,000건 재계산 검증 (약 2분)
python -m src.experiments_country  # 국가별 실험 E1·E4·E5 (한국 E2·E3·E6은 데이터 확보 후), 약 10분
python -m pytest -q                # 단위·화면 테스트 (99개 + 실데이터 있을 때 1개)
streamlit run app_v2.py            # 확장 대시보드 (화면 1~6)
```

- 결과 보고서: [docs/결과보고서_v2.md](docs/결과보고서_v2.md) · 검증 결과: [docs/검증결과_v2.md](docs/검증결과_v2.md)

| 경로 | 내용 |
|---|---|
| `src/schema.py` | 공통 거래 스키마·검증·가명화 |
| `src/sources/sparkov.py` | Sparkov → `data/research/us/transactions.parquet` (소유자 개인정보 형태 열 제거) |
| `src/sources/personal.py` | 개인 거래 파일 읽기·열 자동 매핑·형식 정규화·중복 탐지 |
| `src/privacy.py` | 카드번호(Luhn)·주민번호·민감 열 탐지 |
| `src/store.py` | 개인 거래 SQLite 저장소 (거래, 업로드 기록, 사용자 확인 결과) |
| `config/column_synonyms.yaml` | 카드 이용내역 열 이름 동의어 사전 |
| `src/sources/aihub_card.py` | AI-Hub 카드거래 → 공통 스키마, 품질 보고서, 분기 단위 분할 |
| `config/aihub_codebook.yaml` | AI-Hub 코드 의미표 (데이터 설명서 확인 전까지 비워 둠) |
| `src/features_ext.py` | 신규 특징: 직전 시간대·같은 날 건수, 같은 가맹점 반복, 거리·속도·단시간 원거리, 지역 변화 |
| `src/build_features_v2.py` | 출처별 특징 파일 생성과 재계산 검증 → `outputs/v2/feature_checks_v2.csv` |
| `src/experiments_country.py` | 국가별 실험 → `outputs/v2/e1_*.csv`, `e4_*`, `e5_*`, `cost_thresholds_v2.csv`, `experiment_status.csv` |
| `src/risk.py` | 개인 거래 위험도: 점수·위험 수준·기여 요인·규칙 기반 확인 항목·분포 점검 |
| `views/analysis.py`, `views/performance.py` | 화면 4(이상거래 분석)·5(모델 성능 비교) |
| `app_v2.py`, `views/` | 확장 대시보드 화면 |
| `tests/test_views.py` | 화면 1~6 자동 테스트 (가짜 데이터 임시 프로젝트, AppTest) |

- **개인 거래 저장 위치**: `%LOCALAPPDATA%\CardFDS\personal.db`. 프로젝트(OneDrive) 밖에 있어 클라우드로 동기화되지 않습니다. 폴더는 처음 만들 때 현재 사용자 계정만 접근할 수 있게 권한을 설정합니다. 위치는 환경변수 `FDS_PERSONAL_DIR`로 바꿀 수 있습니다.
- **한국 AI-Hub 데이터**: AI-Hub에 로그인해 「이상 판별을 위한 금융거래 정보 및 사용자 패턴 합성데이터」를 신청한 뒤(내국인만 가능), 카드거래 CSV와 데이터 설명서를 `data/research/kr/raw/`에 넣습니다. 재배포하지 않습니다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `src/features.py` | 필수 열 점검, 시간순 분할, 특징 6개 계산 |
| `src/build_data.py` | 분할·특징 생성, 경계 검증 → `data/features.parquet` |
| `src/experiments.py` | 모든 실험과 임계값 선택, 모델 저장 |
| `app.py` | Streamlit 대시보드 (거래 입력·위험도·모델 비교·임계값 분석·실험 결과) |
| `outputs/*.csv` | 결과 근거 파일 |
| `models/` | 학습된 3개 모델, 고정 임계값 9개, 학습 구간 전체 통계 |
| `data/sample_upload.csv` | 대시보드 업로드 시연용 (원본 Test 중 8개 카드 4,998건) |

## 구현 기준

- **분할**: Train을 시간순 80/10/10. 경계에 같은 시각이 걸리면 뒤로 민다. 원본 Test는 Train 뒤에 이어 붙여 카드 이력을 연속으로 계산한다.
- **이력**: 같은 카드의 *거래 시각보다 앞선* 거래만 사용. 같은 카드·같은 시각 거래는 서로를 이력으로 세지 않는다.
- **특징**: `amt_ratio`, `count_5min/15min/60min`(직전 [t−w, t) 구간), `hour_rarity`, `insufficient_history`(과거 5건 미만). 이력이 부족하면 금액 중앙값과 시간대 비율을 학습 구간 종료 시점에 고정한 전체 통계로 대체한다. 반복 횟수는 항상 실제 이력을 사용한다.
- **검증**: 무작위 2,000건을 원시 이력으로 다시 계산해 2,000건 모두 일치 (`outputs/feature_checks.csv`).
- **모델**: LightGBM(500 트리, lr 0.05, 31 leaves), XGBoost(500 트리, depth 6), Logistic Regression(표준화). 실험 A~D는 양성 가중치 10.
- **임계값**: 검증 구간에서 총비용 `FP + r·FN`이 최소인 점수를 선택하고, 비용이 같으면 경고가 적은 값. 원본 Test에 고정 적용.

## 결과

### 데이터 분할 (`split_summary.csv`)

| 구간 | 용도 | 거래 | 사기 | 정상 |
|---|---|---:|---:|---:|
| Train 앞 80% | 모델 학습 | 1,037,340 | 5,968 | 1,031,372 |
| Train 다음 10% | 보정·임계값 선택 | 129,667 | 706 | 128,961 |
| Train 마지막 10% | 내부 성능 비교 | 129,668 | 832 | 128,836 |
| 원본 Test | 최종 평가 | 555,719 | 2,145 | 553,574 |

필수 열 누락·중복 거래 없음. 학습 구간 정상/사기 비율 172.817.

### LightGBM 가중치와 Platt 보정 (`weight_calibration.csv`)

| scale_pos_weight | AP (내부 비교) | Brier 전→후 | LogLoss 전→후 |
|---|---:|---:|---:|
| 1 | 0.4501 | 0.0032 → 0.0032 | 0.0153 → 0.0155 |
| 5 | 0.4485 | 0.0046 → 0.0032 | 0.0219 → 0.0157 |
| 10 | 0.4455 | 0.0067 → 0.0032 | 0.0312 → 0.0157 |
| 172.817 | 0.4448 | 0.0533 → 0.0033 | 0.1895 → 0.0161 |

가중치는 순위(AP)를 거의 바꾸지 않고 점수를 위로 부풀린다. Platt 보정은 가중치 5 이상에서 Brier·LogLoss를 크게 줄였고, 가중치 1은 이미 보정된 상태라 효과가 없다.

### A. 모델 비교 (내부 비교 구간, 임계값 0.5)

| 모델 | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LightGBM | 0.4455 | 0.337 | 0.555 | 0.419 |
| XGBoost | 0.4450 | 0.331 | 0.538 | 0.410 |
| Logistic Regression | 0.1977 | 0.355 | 0.379 | 0.366 |

원본 Test AP: LightGBM 0.3695, XGBoost 0.3739, LR 0.1220 (`test_model_metrics.csv`).

### B·C. 특징군 단독·제거 (LightGBM, `feature_single.csv` · `feature_ablation.csv`)

| 특징군 | 단독 AP | 제거 후 AP | AP 하락 |
|---|---:|---:|---:|
| 고액 결제 | 0.1968 | 0.1124 | **0.3331** |
| 반복 결제 | 0.0263 | 0.3781 | 0.0674 |
| 이상 시간대 | 0.0543 | 0.3251 | 0.1205 |

전체 특징 AP 0.4455, 무작위 기준(사기 비율) 0.0064. 고액 결제가 가장 크게 기여한다.

### D. 비용별 고정 임계값 → 원본 Test (`test_threshold_comparison.csv`)

| 모델 | 비용 | 임계값 | 총비용 0.5 → 고정 | 변화 |
|---|---|---:|---:|---:|
| LightGBM | 5:1 | 0.692 | 8,650 → 7,690 | −11.1% |
| LightGBM | 20:1 | 0.314 | 22,915 → 22,929 | +0.1% |
| LightGBM | 100:1 | 0.070 | 98,995 → 80,610 | −18.6% |
| XGBoost | 5:1 | 0.647 | 8,591 → 7,674 | −10.7% |
| XGBoost | 20:1 | 0.281 | 22,946 → 22,750 | −0.9% |
| XGBoost | 100:1 | 0.078 | 99,506 → 77,780 | −21.8% |
| LR | 5:1 | 0.361 | 9,239 → 9,566 | +3.5% |
| LR | 20:1 | 0.159 | 29,384 → 28,558 | −2.8% |
| LR | 100:1 | 0.060 | 136,824 → 101,422 | −25.9% |

9개 조합 중 7개에서 총비용 감소. 20:1에서는 기본값 0.5가 이미 최적에 가까워 효과가 거의 없고, LightGBM은 0.1% 증가했다.

## 해석 범위

Sparkov는 모의 데이터이며 비용은 가정한 상대 단위다. 결과는 주어진 비용 조건에서 탐지 정책을 비교하는 데 쓴다. 경고 건수와 실제 피해액은 구분해 해석한다.
