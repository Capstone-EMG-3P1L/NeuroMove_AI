import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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


def _compute_rest_baseline(
    df: pd.DataFrame,
) -> dict[int, tuple[float, float]]:
    """
    [수정] REST 라벨 행에서 채널별 mean/std 계산 → 학습용 baseline
    세션 추론 시 백엔드가 주는 calibration.baseline과 동일한 역할

    학습: 이 함수로 REST baseline 계산 → z-score
    추론: 백엔드가 준 calibration.baseline → apply_calibration_baseline → z-score
    """
    # REST 또는 NEUTRAL 라벨을 baseline으로 사용
    rest_mask = df[LABEL_COLUMN].astype(str).str.upper().isin(["REST", "NEUTRAL"])
    rest_df = df[rest_mask]

    if rest_df.empty:
        raise ValueError("REST 라벨 데이터가 없어서 baseline 계산 불가")

    baseline = {}
    for ch_idx in range(CHANNEL_COUNT):
        ch_cols = _get_channel_columns(df, ch_idx)
        rest_samples = rest_df[ch_cols].to_numpy(dtype=float).flatten()
        baseline[ch_idx] = (float(np.mean(rest_samples)), float(np.std(rest_samples)))

    return baseline


def _apply_zscore_rectify(
    samples: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
) -> np.ndarray:
    """
    [수정] calibration 기반 z-score + 정류
    추론 시 signal_processing_service.apply_calibration_baseline + rectify_signal 과 동일한 처리

    (sample - REST_mean) / REST_std → |result|
    """
    if baseline_std > 0:
        processed = (samples - baseline_mean) / baseline_std
    else:
        processed = samples - baseline_mean

    return np.abs(processed)


def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    [수정] REST baseline 기반 z-score + 정류 후 feature 추출
    추론 파이프라인(calibration 기반 z-score)과 동일한 전처리
    """
    # 1. REST 행에서 채널별 baseline(mean/std) 계산
    baseline = _compute_rest_baseline(df)

    print("=== REST Baseline (학습용) ===")
    for ch_idx, (mean, std) in baseline.items():
        print(f"  ch{ch_idx}: mean={mean:.2f}, std={std:.2f}")
    print()

    feature_df = pd.DataFrame()

    for channel_index in range(CHANNEL_COUNT):
        channel_columns = _get_channel_columns(df, channel_index)
        samples = df[channel_columns].to_numpy(dtype=float)

        # [수정] REST baseline 기반 z-score + 정류
        mean, std = baseline[channel_index]
        samples = _apply_zscore_rectify(samples, mean, std)

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