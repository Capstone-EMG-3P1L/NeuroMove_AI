import os
import pickle
from functools import lru_cache

from schemas.stream_schema import InferenceResultSchema


MODEL_PATH = "models/intent_model.pkl"
MODEL_VERSION = "mock-v5"

# 현재 프로젝트에서 사용하는 intent 라벨
# LEFT  : 왼쪽으로 고개 돌림
# RIGHT : 오른쪽으로 고개 돌림
# REST  : 힘 빼고 가만히 있음 / 알 수 없는 신호
# STOP  : 턱에 부착한 stop 전용 채널 활성
VALID_INTENTS = {"LEFT", "RIGHT", "REST", "STOP"}

# 3채널 기준 feature vector:
# [ch0_mav, ch0_rms, ch1_mav, ch1_rms, ch2_mav, ch2_rms]
STOP_CHANNEL_MAV_INDEX = 4
STOP_CHANNEL_RMS_INDEX = 5

# calibration에 STOP threshold가 없을 때 사용할 fallback 값
DEFAULT_STOP_THRESHOLD = 2.0


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


def _is_stop_channel_active(
    feature_vector: list[float],
    calibration,
) -> bool:
    """
    턱에 붙인 STOP 전용 채널이 활성화됐는지 확인한다.

    3채널 기준 feature vector:
    [ch0_mav, ch0_rms, ch1_mav, ch1_rms, ch2_mav, ch2_rms]

    ch2가 STOP 전용 채널이므로 ch2_mav 또는 ch2_rms가
    STOP threshold 이상이면 STOP으로 판단한다.
    """
    if len(feature_vector) <= STOP_CHANNEL_RMS_INDEX:
        return False

    stop_threshold = _get_intent_threshold(calibration, "STOP")

    if stop_threshold is None:
        stop_threshold = DEFAULT_STOP_THRESHOLD

    stop_mav = feature_vector[STOP_CHANNEL_MAV_INDEX]
    stop_rms = feature_vector[STOP_CHANNEL_RMS_INDEX]

    return stop_mav >= stop_threshold or stop_rms >= stop_threshold


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

    # STOP 전용 채널이 활성화되면 우선 STOP 처리
    if _is_stop_channel_active(feature_vector, calibration):
        stop_confidence = max(
            feature_vector[STOP_CHANNEL_MAV_INDEX],
            feature_vector[STOP_CHANNEL_RMS_INDEX],
        )
        return "STOP", round(min(stop_confidence, 1.0), 4)

    channel_scores = []

    # 채널별 MAV/RMS 평균 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    # 방향 판단은 ch0, ch1만 사용
    direction_scores = channel_scores[:2]

    if not direction_scores:
        return "REST", 0.0

    max_score = max(direction_scores)

    # 방향 신호가 약하면 REST
    if max_score < 0.15:
        return "REST", round(max_score, 4)

    max_channel = direction_scores.index(max_score)

    intent_map = {
        0: "LEFT",
        1: "RIGHT",
    }

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

    # STOP 전용 채널이 활성화되면 모델보다 우선해서 STOP 처리
    if _is_stop_channel_active(feature_vector, calibration):
        stop_confidence = max(
            feature_vector[STOP_CHANNEL_MAV_INDEX],
            feature_vector[STOP_CHANNEL_RMS_INDEX],
        )

        return InferenceResultSchema(
            predicted_intent="STOP",
            confidence=round(min(stop_confidence, 1.0), 4),
            feature_vector=feature_vector,
            model_version="stop-threshold-rule",
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

        # prediction, confidence = _apply_intent_threshold(
        #     prediction,
        #     confidence,
        #     calibration,
        # )

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