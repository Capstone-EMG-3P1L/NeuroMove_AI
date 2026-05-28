import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.signal_processing_service import remove_dc_offset, rectify_signal


# raw EMG window 데이터 폴더 경로
DATA_DIR = "training/data"

# 학습된 모델 저장 경로
MODEL_PATH = "models/intent_model.pkl"

# 사용할 채널 수
CHANNEL_COUNT = 3

# 최종 feature 컬럼
FEATURE_COLUMNS = [
    "ch0_mav",
    "ch0_rms",
    "ch1_mav",
    "ch1_rms",
    "ch2_mav",
    "ch2_rms",
]

# 정답 라벨 컬럼
LABEL_COLUMN = "label"

# 허용하는 intent 라벨
VALID_LABELS = {"LEFT", "RIGHT", "REST", "STOP"}

# raw 데이터 라벨을 프로젝트 intent 라벨로 변환
LABEL_MAP = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "REST": "REST",
    "STOP": "STOP",
    "NEUTRAL": "REST",
    "CLENCH": "STOP",
}


def load_training_data() -> pd.DataFrame:
    # training/data 안의 모든 CSV 파일을 읽어서 하나로 합치기
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Training data directory not found: {DATA_DIR}\n"
            "먼저 training/data 디렉토리 생성 필요"
        )

    csv_files = [
        file for file in os.listdir(DATA_DIR)
        if file.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    df_list = []

    for file in sorted(csv_files):
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

        print(f"Loaded: {file_path} ({len(df)} rows)")

    merged_df = pd.concat(df_list, ignore_index=True)

    print(f"Total training rows: {len(merged_df)}")

    return merged_df


def _get_channel_columns(df: pd.DataFrame, channel_index: int) -> list[str]:
    prefix = f"ch{channel_index}_"

    channel_columns = [
        col for col in df.columns
        if col.startswith(prefix)
    ]

    if not channel_columns:
        raise ValueError(f"channel {channel_index} sample columns not found")

    # ch0_0, ch0_1, ..., ch0_31 순서 보장
    channel_columns.sort(key=lambda col: int(col.split("_")[1]))

    return channel_columns


def _calculate_mav(samples: np.ndarray) -> np.ndarray:
    # window별 MAV 계산
    return np.mean(np.abs(samples), axis=1)


def _calculate_rms(samples: np.ndarray) -> np.ndarray:
    # window별 RMS 계산
    return np.sqrt(np.mean(np.square(samples), axis=1))


def _map_labels(labels: pd.Series) -> pd.Series:
    # raw label을 프로젝트 intent 라벨로 변환
    mapped_labels = labels.astype(str).str.upper().map(LABEL_MAP)

    if mapped_labels.isnull().any():
        invalid_raw_labels = set(
            labels[mapped_labels.isnull()].astype(str).str.upper().unique()
        )
        raise ValueError(f"Invalid raw labels found: {invalid_raw_labels}")

    return mapped_labels


def _preprocess_samples(samples: np.ndarray) -> np.ndarray:
    """
    학습 데이터에도 실시간 추론과 동일한 신호처리 적용
    정류만 적용
    DC 제거는 절대 amplitude 차이를 보존하기 위해 적용하지 않음
    """
    processed = np.zeros_like(samples)

    for i in range(samples.shape[0]):
        signal = samples[i]
        # DC 제거하지 않음
        # signal = remove_dc_offset(signal)
        signal = rectify_signal(signal)
        processed[i] = signal

    return processed


def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    # raw sample 컬럼을 MAV/RMS feature 컬럼으로 변환
    feature_df = pd.DataFrame()

    for channel_index in range(CHANNEL_COUNT):
        channel_columns = _get_channel_columns(df, channel_index)
        samples = df[channel_columns].to_numpy(dtype=float)

        # 신호처리(DC 제거 + 정류)를 먼저 적용한 뒤 feature 추출
        samples = _preprocess_samples(samples)

        feature_df[f"ch{channel_index}_mav"] = _calculate_mav(samples)
        feature_df[f"ch{channel_index}_rms"] = _calculate_rms(samples)

    feature_df[LABEL_COLUMN] = _map_labels(df[LABEL_COLUMN])

    return feature_df


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
    raw_df = load_training_data()

    feature_df = extract_features_from_raw(raw_df)
    validate_training_data(feature_df)

    model = train_model(feature_df)
    save_model(model)


if __name__ == "__main__":
    main()