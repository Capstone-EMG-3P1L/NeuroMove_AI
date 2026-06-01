"""
mock_emg_client.py — 시연용 Mock EMG WebSocket 클라이언트

역할: EMG 보드 대신 WebSocket으로 EMG 데이터만 계속 전송
세션 시작/종료는 백엔드가 AI 서버에 요청 (이 스크립트는 관여하지 않음)

테스트 흐름:
    1. 백엔드에서 AI 서버로 세션 시작 요청 (calibration 포함)
    2. python mock_emg_client.py 실행 → EMG 데이터 WebSocket 전송
    3. AI 서버가 신호처리 + 추론 → 백엔드 /api/ai/intent 로 POST
    4. 백엔드에서 AI 서버로 세션 종료 요청
    5. Ctrl+C로 이 스크립트 종료

사용법:
    python mock_emg_client.py                # 기본 실행
    python mock_emg_client.py --no-failsafe  # fail-safe 테스트 생략
"""

import argparse
import asyncio
import json
import random
import time

import websockets


# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
WS_URL = "wss://yeonwoo.shop/ai/stream/ws"
DEVICE_ID = "emg-esp32-A12F"

SAMPLING_RATE = 100
WINDOW_SIZE = 32
CHANNEL_COUNT = 3
SEND_INTERVAL = 0.32  # 초 (windowSize / samplingRate)

# 2초 유지 = 약 6개 window
WINDOWS_PER_2SEC = 6


# ══════════════════════════════════════════════════════════════
# EMG 신호 파라미터 (emg_features_04.csv 실측 통계 기반)
# ══════════════════════════════════════════════════════════════
EMG_PARAMS = {
    "REST": {
        0: (2700, 72),
        1: (2699, 76),
        2: (2696, 111),
    },
    "LEFT": {
        0: (3045, 180),  # ch0 활성
        1: (2846, 128),
        2: (2697, 104),
    },
    "RIGHT": {
        0: (2852, 100),
        1: (3053, 181),  # ch1 활성
        2: (2700, 84),
    },
    "STOP": {
        0: (2892, 285),
        1: (2894, 275),
        2: (3243, 266),  # ch2 활성
    },
}


# ══════════════════════════════════════════════════════════════
# EMG 데이터 생성
# ══════════════════════════════════════════════════════════════
def generate_emg_samples(
    intent: str,
    channel_index: int,
    noise_multiplier: float = 1.0,
    amplitude_scale: float = 1.0,
) -> list[int]:
    params = EMG_PARAMS.get(intent, EMG_PARAMS["REST"])
    base_mean, base_std = params.get(channel_index, (2700, 80))

    rest_mean = EMG_PARAMS["REST"][channel_index][0]

    activation = base_mean - rest_mean
    mean = rest_mean + activation * amplitude_scale
    std = base_std * noise_multiplier

    samples = []
    for _ in range(WINDOW_SIZE):
        value = int(mean + random.gauss(0, std))
        value = max(0, min(value, 4095))
        samples.append(value)

    return samples


def build_emg_message(
    sequence_number: int,
    intent: str = "REST",
    timestamp_ms: int | None = None,
    noise_multiplier: float = 1.0,
    amplitude_scale: float = 1.0,
) -> dict:
    channels = []
    for ch_idx in range(CHANNEL_COUNT):
        channels.append({
            "channelIndex": ch_idx,
            "samples": generate_emg_samples(
                intent, ch_idx,
                noise_multiplier=noise_multiplier,
                amplitude_scale=amplitude_scale,
            ),
        })

    return {
        "deviceId": DEVICE_ID,
        "sequenceNumber": sequence_number,
        "timestamp": timestamp_ms if timestamp_ms is not None else int(time.time() * 1000),
        "samplingRate": SAMPLING_RATE,
        "windowSize": WINDOW_SIZE,
        "channels": channels,
    }


# ══════════════════════════════════════════════════════════════
# 시연 시나리오
# ══════════════════════════════════════════════════════════════
# (intent, window수, 설명, noise_multiplier, amplitude_scale)

DEMO_SCENARIO = [
    # ── 정상 운행 (충분한 정상 구간) ──
    ("REST",  WINDOWS_PER_2SEC,  "정상 REST",       1.0, 1.0),
    ("LEFT",  WINDOWS_PER_2SEC,  "정상 LEFT 2초",   1.0, 1.0),
    ("REST",  3,                 "전환 REST",       1.0, 1.0),
    ("RIGHT", WINDOWS_PER_2SEC,  "정상 RIGHT 2초",  1.0, 1.0),
    ("REST",  3,                 "전환 REST",       1.0, 1.0),
    ("STOP",  WINDOWS_PER_2SEC,  "정상 STOP 2초",   1.0, 1.0),
    ("REST",  WINDOWS_PER_2SEC,  "정상 REST 회복",  1.0, 1.0),

    # ── 피로도 점진 상승 (완만하게) ──
    ("LEFT",  WINDOWS_PER_2SEC,  "피로 LEFT 85%",   1.0, 0.85),
    ("REST",  3,                 "전환",            1.0, 1.0),
    ("RIGHT", WINDOWS_PER_2SEC,  "피로 RIGHT 70%",  1.0, 0.7),
    ("REST",  WINDOWS_PER_2SEC,  "회복 REST",       1.0, 1.0),
    ("LEFT",  WINDOWS_PER_2SEC,  "피로 LEFT 60%",   1.0, 0.6),
    ("REST",  WINDOWS_PER_2SEC,  "회복 REST",       1.0, 1.0),

    # ── 신호 불안정 (노이즈 적당히) ──
    ("LEFT",  WINDOWS_PER_2SEC,  "불안정 LEFT",     1.8, 1.0),
    ("REST",  WINDOWS_PER_2SEC,  "안정 REST",       1.0, 1.0),
    ("RIGHT", WINDOWS_PER_2SEC,  "불안정 RIGHT",    1.8, 1.0),
    ("REST",  WINDOWS_PER_2SEC,  "안정 REST",       1.0, 1.0),

    # ── 고위험 (피로 + 노이즈, 하지만 극단값 아님) ──
    ("LEFT",  WINDOWS_PER_2SEC,  "고위험 LEFT",     1.5, 0.6),
    ("REST",  WINDOWS_PER_2SEC,  "회복 REST",       1.0, 1.0),

    # ── 정상 복귀 ──
    ("LEFT",  WINDOWS_PER_2SEC,  "복귀 LEFT",       1.0, 1.0),
    ("RIGHT", WINDOWS_PER_2SEC,  "복귀 RIGHT",      1.0, 1.0),
    ("REST",  WINDOWS_PER_2SEC,  "최종 REST",       1.0, 1.0),
]


# ══════════════════════════════════════════════════════════════
# WebSocket EMG 전송
# ══════════════════════════════════════════════════════════════
async def run(include_failsafe: bool):
    print(f"  WebSocket 연결 중: {WS_URL}")

    async with websockets.connect(WS_URL) as ws:
        print(f"  연결 성공! EMG 데이터 전송 시작\n")

        seq = 0
        cycle = 0

        while True:
            cycle += 1
            print(f"{'═' * 50}")
            print(f"  Cycle {cycle}")
            print(f"{'═' * 50}")

            for intent, count, desc, noise_mul, amp_scale in DEMO_SCENARIO:
                risk_tag = " ⚠ HIGH RISK" if (noise_mul > 1.0 or amp_scale < 1.0) else ""
                print(f"\n  [{desc}] {intent} x{count}{risk_tag}")

                for _ in range(count):
                    msg = build_emg_message(
                        sequence_number=seq,
                        intent=intent,
                        noise_multiplier=noise_mul,
                        amplitude_scale=amp_scale,
                    )
                    await ws.send(json.dumps(msg))

                    try:
                        ack = await asyncio.wait_for(ws.recv(), timeout=5)
                        ack_data = json.loads(ack)
                        status = "OK" if ack_data.get("success") else "FAIL"
                        print(f"    seq={seq:04d}  {intent:>5s}  {status}  {ack_data.get('message', '')}")
                    except asyncio.TimeoutError:
                        print(f"    seq={seq:04d}  {intent:>5s}  TIMEOUT")

                    seq += 1
                    await asyncio.sleep(SEND_INTERVAL)

            # ── Fail-safe 테스트 (첫 cycle만) ──
            if include_failsafe:
                print(f"\n{'─' * 50}")
                print(f"  Fail-safe Tests")
                print(f"{'─' * 50}")

                # 1. 중복 시퀀스
                dup_seq = seq - 1
                print(f"\n  [중복 시퀀스] seq={dup_seq} 재전송 → 400 예상")
                msg = build_emg_message(sequence_number=dup_seq, intent="LEFT")
                await ws.send(json.dumps(msg))
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=5)
                    ack_data = json.loads(ack)
                    print(f"    결과: success={ack_data.get('success')}  {ack_data.get('message', '')}")
                except asyncio.TimeoutError:
                    print(f"    결과: TIMEOUT")
                await asyncio.sleep(SEND_INTERVAL)

                # 2. 3초 초과 타임스탬프
                old_ts = int(time.time() * 1000) - 5000
                print(f"\n  [3초 초과 타임스탬프] 5초 전 timestamp → BLOCKED 예상")
                msg = build_emg_message(sequence_number=seq, intent="RIGHT", timestamp_ms=old_ts)
                await ws.send(json.dumps(msg))
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=5)
                    ack_data = json.loads(ack)
                    print(f"    결과: success={ack_data.get('success')}  {ack_data.get('message', '')}")
                except asyncio.TimeoutError:
                    print(f"    결과: TIMEOUT")
                seq += 1
                await asyncio.sleep(SEND_INTERVAL)

                # 3. 모호한 신호
                print(f"\n  [모호한 신호] 모든 채널 동일 → confidence 낮음 예상")
                for _ in range(WINDOWS_PER_2SEC):
                    msg = build_emg_message(sequence_number=seq, intent="REST", noise_multiplier=0.3)
                    await ws.send(json.dumps(msg))
                    try:
                        ack = await asyncio.wait_for(ws.recv(), timeout=5)
                        ack_data = json.loads(ack)
                        print(f"    seq={seq:04d}  REST(모호)  {ack_data.get('message', '')}")
                    except asyncio.TimeoutError:
                        print(f"    seq={seq:04d}  REST(모호)  TIMEOUT")
                    seq += 1
                    await asyncio.sleep(SEND_INTERVAL)

                include_failsafe = False
                print(f"\n  Fail-safe 완료.\n")

            print(f"\n  Cycle {cycle} 끝. 반복 중... (Ctrl+C 종료)\n")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NeuroMove Mock EMG Client")
    parser.add_argument("--no-failsafe", action="store_true", help="fail-safe 테스트 생략")
    args = parser.parse_args()

    print("=" * 50)
    print("  NeuroMove Mock EMG Client")
    print("=" * 50)
    print(f"  WebSocket : {WS_URL}")
    print(f"  Device    : {DEVICE_ID}")
    print(f"  Interval  : {SEND_INTERVAL}s")
    print(f"  Fail-safe : {'생략' if args.no_failsafe else '포함'}")
    print("=" * 50)
    print()
    print("  ※ 세션 시작/종료는 백엔드에서 처리")
    print("  ※ 백엔드가 세션 시작한 뒤 이 스크립트 실행")
    print()

    try:
        asyncio.run(run(include_failsafe=not args.no_failsafe))
    except KeyboardInterrupt:
        print("\n\n  Ctrl+C — 종료")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n  WebSocket 연결 끊김: {e}")
    except Exception as e:
        print(f"\n  오류: {e}")


if __name__ == "__main__":
    main()
