# Pokemon Classifier with Transfer Learning

포켓몬 이미지 150종을 분류하는 Transfer Learning 기반 이미지 분류기.

## 실험 설정 및 결과

| 실험 | Backbone | Fine-tuning 범위 | Pretrained | Acc | Recall | Precision | F1 | Top-5 Acc |
|------|----------|-----------------|------------|-----|--------|-----------|-----|-----------|
| EXP-01 | ResNet-50 | Head only | O | - | - | - | - | - |
| EXP-02 | ResNet-50 | Full FT | O | - | - | - | - | - |
| EXP-03 | EfficientNet-B0 | Last 2 blocks | O | - | - | - | - | - |
| EXP-04 | ViT-B/16 | Full FT | O | - | - | - | - | - |
| EXP-05 | ResNet-50 | Full FT | X (scratch) | - | - | - | - | - |

> 실험 실행 후 `results/comparison_table.csv` 에서 수치 확인 가능

## Learning Curves

학습 완료 후 `results/EXP-XX/learning_curve.png` 참조.

## 예측 예시

학습 완료 후 `results/EXP-XX/error_samples.png` 참조.

## 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터셋 다운로드 (Kaggle CLI)
kaggle datasets download -d lantian773030/pokemonclassification
unzip pokemonclassification.zip -d data/

# 3. 전체 실험 실행 (약 2~5시간, GPU 권장)
python src/experiment.py

# 특정 실험만 실행
python src/experiment.py --exp_ids EXP-01 EXP-02

# 4. Streamlit 데모 실행
streamlit run app.py
# → http://localhost:8501
```

## 데이터셋

- [Kaggle: 7,000 Labeled Pokemon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- 150개 클래스, 약 7,000장 이미지
- 분할: Train 70% / Val 15% / Test 15% (Stratified)

## 프로젝트 구조

```
pokemon-transfer-learning-classifier/
├── data/PokemonData/        # Kaggle 데이터셋
├── models/                  # 학습된 .pth 파일
├── results/                 # 실험별 지표, 시각화
├── src/
│   ├── dataset.py           # 데이터 로딩 및 전처리
│   ├── model.py             # 모델 팩토리 (backbone 선택)
│   ├── train.py             # 학습 루프
│   ├── evaluate.py          # 평가 및 시각화
│   └── experiment.py        # 실험 정의 및 일괄 실행
├── app.py                   # Streamlit 데모
└── requirements.txt
```
