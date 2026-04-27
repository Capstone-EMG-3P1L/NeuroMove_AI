import os
import pickle
from functools import lru_cache

from schemas.stream_schema import InferenceResultSchema


MODEL_PATH = "models/intent_model.pkl"
MODEL_VERSION = "mock-v4"

# 현재 프로젝트에서 사용하는 intent 라벨
# LEFT  : 왼쪽으로 고개 돌림
# RIGHT : 오른쪽으로 고개 돌림
# REST  : 힘 빼고 가만히 있음 / 알 수 없는 신호
# STOP  : 힘줘서 멈춤
VALID_INTENTS = {"LEFT", "RIGHT", "REST", "STOP"}


@lru_cache(maxsize=1)
def load_model():
    # 모델 파일이 있으면 한 번만 로드해서 캐싱
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    return None


def _get_activation_threshold(calibration) -> float | None:
    if calibration is None:
        return None

    threshold = getattr(calibration, "activationThreshold", None)
    if threshold is None:
        return None

    return float(threshold)


def _get_intent_threshold(calibration, intent: str) -> float | None:
    if calibration is None:
        return None

    thresholds = getattr(calibration, "intentThresholds", None)
    if not thresholds:
        return None

    # intentThresholds의 key가 Enum일 수도 있고 문자열일 수도 있어서 둘 다 처리
    if intent in thresholds:
        return float(thresholds[intent])

    for key, value in thresholds.items():
        key_value = getattr(key, "value", key)
        if key_value == intent:
            return float(value)

    return None


def _is_below_activation_threshold(
    feature_vector: list[float],
    calibration,
) -> bool:
    threshold = _get_activation_threshold(calibration)

    if threshold is None:
        return False

    if not feature_vector:
        return True

    activation_score = sum(feature_vector) / len(feature_vector)

    return activation_score < threshold


def _apply_intent_threshold(
    intent: str,
    confidence: float,
    calibration,
) -> tuple[str, float]:
    threshold = _get_intent_threshold(calibration, intent)

    if threshold is None:
        return intent, confidence

    if confidence < threshold:
        return "REST", confidence

    return intent, confidence


def _mock_predict(
    feature_vector: list[float],
    calibration=None,
) -> tuple[str, float]:
    # feature가 없으면 REST
    if not feature_vector:
        return "REST", 0.0

    # activationThreshold보다 약하면 REST
    if _is_below_activation_threshold(feature_vector, calibration):
        return "REST", 0.0

    channel_scores = []

    # 채널별 MAV/RMS 평균 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    max_score = max(channel_scores)

    # 전체 신호가 약하면 REST
    if max_score < 0.15:
        return "REST", round(max_score, 4)

    # 전체적으로 신호가 강하면 STOP으로 처리
    avg_score = sum(channel_scores) / len(channel_scores)

    if avg_score >= 0.5:
        intent = "STOP"
        confidence = round(min(avg_score, 1.0), 4)
        return _apply_intent_threshold(intent, confidence, calibration)

    max_channel = channel_scores.index(max_score)

    # 현재는 좌/우 고개 방향만 사용
    intent_map = {
        0: "LEFT",
        1: "RIGHT",
    }

    # 매핑되지 않는 채널이면 REST 처리
    intent = intent_map.get(max_channel, "REST")
    confidence = round(min(max_score, 1.0), 4)

    return _apply_intent_threshold(intent, confidence, calibration)


def predict_intent(
    feature_vector: list[float],
    calibration=None,
) -> InferenceResultSchema:
    # activationThreshold보다 약하면 모델 예측 전에 REST 처리
    if _is_below_activation_threshold(feature_vector, calibration):
        return InferenceResultSchema(
            predicted_intent="REST",
            confidence=0.0,
            feature_vector=feature_vector,
            model_version="threshold-rule",
        )

    model = load_model()

    if model is not None:
        prediction = str(model.predict([feature_vector])[0])

        # 모델이 허용되지 않은 라벨을 예측하면 REST 처리
        if prediction not in VALID_INTENTS:
            prediction = "REST"

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([feature_vector])[0]
            confidence = round(float(max(probabilities)), 4)
        else:
            confidence = 0.8

        prediction, confidence = _apply_intent_threshold(
            prediction,
            confidence,
            calibration,
        )

        return InferenceResultSchema(
            predicted_intent=prediction,
            confidence=confidence,
            feature_vector=feature_vector,
            model_version="trained-model",
        )

    intent, confidence = _mock_predict(feature_vector, calibration)

    return InferenceResultSchema(
        predicted_intent=intent,
        confidence=confidence,
        feature_vector=feature_vector,
        model_version=MODEL_VERSION,
    )