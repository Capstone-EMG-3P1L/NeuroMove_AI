import os
import pickle
from functools import lru_cache

from schemas.stream_schema import InferenceResultSchema


MODEL_PATH = "models/intent_model.pkl"
MODEL_VERSION = "mock-v2"

VALID_INTENTS = {"LEFT", "RIGHT", "FORWARD", "BACKWARD", "STOP", "UNKNOWN"}


@lru_cache(maxsize=1)
def load_model():
    # 모델 파일이 있으면 한 번만 로드해서 캐싱
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    return None


def _mock_predict(feature_vector: list[float]) -> tuple[str, float]:
    # feature가 없으면 UNKNOWN
    if not feature_vector:
        return "UNKNOWN", 0.0

    channel_scores = []

    # 채널별 MAV/RMS 평균 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    max_score = max(channel_scores)

    # 약한 신호는 STOP
    if max_score < 0.15:
        return "STOP", round(max_score, 4)

    max_channel = channel_scores.index(max_score)

    intent_map = {
        0: "LEFT",
        1: "RIGHT",
        2: "FORWARD",
        3: "BACKWARD",
    }

    intent = intent_map.get(max_channel, "UNKNOWN")

    if intent == "UNKNOWN":
        return "UNKNOWN", 0.0

    return intent, round(min(max_score, 1.0), 4)


def predict_intent(feature_vector: list[float]) -> InferenceResultSchema:
    model = load_model()

    if model is not None:
        prediction = str(model.predict([feature_vector])[0])

        if prediction not in VALID_INTENTS:
            prediction = "UNKNOWN"

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([feature_vector])[0]
            confidence = round(float(max(probabilities)), 4)
        else:
            confidence = 0.8

        return InferenceResultSchema(
            predicted_intent=prediction,
            confidence=confidence,
            feature_vector=feature_vector,
            model_version="trained-model",
        )

    intent, confidence = _mock_predict(feature_vector)

    return InferenceResultSchema(
        predicted_intent=intent,
        confidence=confidence,
        feature_vector=feature_vector,
        model_version=MODEL_VERSION,
    )