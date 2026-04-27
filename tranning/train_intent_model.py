import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 학습 데이터 경로
DATA_PATH = "training/emg_features.csv"

# 학습된 모델 저장 경로
MODEL_PATH = "models/intent_model.pkl"

# 사용할 feature 컬럼
FEATURE_COLUMNS = [
    "ch0_mav",
    "ch0_rms",
    "ch1_mav",
    "ch1_rms",
    "ch2_mav",
    "ch2_rms",
    "ch3_mav",
    "ch3_rms",
]

# 정답 라벨 컬럼
LABEL_COLUMN = "label"

# 허용하는 intent 라벨
VALID_LABELS = {"LEFT", "RIGHT", "REST", "STOP"}


def load_training_data() -> pd.DataFrame:
    # 학습용 데이터 불러오기
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "먼저 training/emg_features.csv 파일 생성 필요"
        )

    df = pd.read_csv(DATA_PATH)
    return df


def validate_training_data(df: pd.DataFrame) -> None:
    # 학습 데이터 형식이 올바른지 검증
    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in training data: {missing_columns}")

    if df.empty:
        raise ValueError("Training data is empty")

    invalid_labels = set(df[LABEL_COLUMN].unique()) - VALID_LABELS

    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature columns contain missing values")

    if df[LABEL_COLUMN].isnull().any():
        raise ValueError("Label column contains missing values")


def train_model(df: pd.DataFrame) -> RandomForestClassifier:

    # RandomForestClassifier 모델을 학습
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    # 데이터가 너무 적으면 stratify가 실패할 수 있어서 조건 처리
    stratify_option = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify_option,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("=== Training Result ===")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    return model


def save_model(model: RandomForestClassifier) -> None:
    
    # 학습된 모델을 models/intent_model.pkl로 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to: {MODEL_PATH}")


def main() -> None:
    df = load_training_data()
    validate_training_data(df)

    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()