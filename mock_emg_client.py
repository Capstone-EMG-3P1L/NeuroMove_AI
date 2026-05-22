"""
mock_emg_client.py — 시연용 Mock EMG WebSocket 클라이언트

[수정] 시연 시 ESP32 보드 없이 mock 데이터를 AI 서버 WebSocket으로 전송하는 스크립트

사용법:
    python mock_emg_client.py                          # 기본 설정으로 실행
    python mock_emg_client.py --device-id MOCK-001     # 디바이스 ID 지정
    python mock_emg_client.py --csv training/emg_features.csv  # CSV 데이터 사용
    python mock_emg_client.py --mode random             # 랜덤 생성 모드
    python mock_emg_client.py --scenario demo           # 시연 시나리오 (REST→LEFT→RIGHT→STOP 순환)

흐름:
    이 스크립트 ──WebSocket──▶ /ai/stream/ws ──▶ StreamService.handle_emg_window
                                                      │
                                                      ├─ CALIBRATION → 캘리브레이션 버퍼 누적
                                                      └─ SESSION     → 신호처리 → 추론 → 백엔드 전송

주의:
    - 서버에서 session 또는 calibration을 먼저 시작해야 데이터가 처리됨 (IDLE이면 drop)
    - REST API로 calibration/session 시작 후 이 스크립트 실행
"""

import argparse
import asyncio
import csv
import json
import math
import random
import time
from typing import Optional

import websockets


# ── 기본 설정 ──
DEFAULT_WS_URL = "ws://localhost:8000/ai/stream/ws"
DEFAULT_DEVICE_ID = "MOCK-EMG-001"
DEFAULT_SAMPLING_RATE = 100  # Hz
DEFAULT_WINDOW_SIZE = 32     # samples per window
DEFAULT_CHANNEL_COUNT = 3
DEFAULT_SEND_INTERVAL = 0.32  # 초 (windowSize / samplingRate = 32/100)

# ── EMG 시뮬레이션 파라미터 ──
# 실제 EMG 보드 ADC 값 범위 기준 (12-bit ADC, ~2600-3300)
REST_BASELINE = 2700       # 안정 시 기본값
LEFT_BASELINE_CH0 = 3150   # LEFT 동작: ch0(왼쪽 근육)이 강하게 활성
LEFT_BASELINE_CH1 = 2700   # LEFT 동작: ch1(오른쪽 근육)은 안정
RIGHT_BASELINE_CH0 = 2700  # RIGHT 동작: ch0(왼쪽 근육)은 안정
RIGHT_BASELINE_CH1 = 3150  # RIGHT 동작: ch1(오른쪽 근육)이 강하게 활성
STOP_BASELINE_CH2 = 3200   # STOP 동작: ch2(턱 채널)이 활성
NOISE_STD = 20             # 기본 노이즈 표준편차


def generate_emg_samples(
    intent: str,
    channel_index: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[int]:
    """
    intent와 채널에 따라 mock EMG sample 생성
    실제 EMG 보드의 ADC 출력과 유사한 값 범위 사용
    """
    if intent == "LEFT":
        if channel_index == 0:
            baseline = LEFT_BASELINE_CH0
            noise_std = 80  # 활성 채널은 진폭 큼
        elif channel_index == 1:
            baseline = LEFT_BASELINE_CH1
            noise_std = NOISE_STD
        else:
            baseline = REST_BASELINE
            noise_std = NOISE_STD

    elif intent == "RIGHT":
        if channel_index == 0:
            baseline = RIGHT_BASELINE_CH0
            noise_std = NOISE_STD
        elif channel_index == 1:
            baseline = RIGHT_BASELINE_CH1
            noise_std = 80
        else:
            baseline = REST_BASELINE
            noise_std = NOISE_STD

    elif intent == "STOP":
        if channel_index == 2:
            baseline = STOP_BASELINE_CH2
            noise_std = 60
        else:
            baseline = REST_BASELINE
            noise_std = NOISE_STD

    else:  # REST
        baseline = REST_BASELINE
        noise_std = NOISE_STD

    samples = []
    for _ in range(window_size):
        value = int(baseline + random.gauss(0, noise_std))
        value = max(0, min(value, 4095))  # 12-bit ADC range
        samples.append(value)

    return samples


def build_emg_message(
    device_id: str,
    sequence_number: int,
    intent: str = "REST",
    channel_count: int = DEFAULT_CHANNEL_COUNT,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    """EmgWindowMessage 형식의 JSON 메시지 생성"""
    channels = []
    for ch_idx in range(channel_count):
        channels.append({
            "channelIndex": ch_idx,
            "samples": generate_emg_samples(intent, ch_idx, window_size),
        })

    return {
        "deviceId": device_id,
        "sequenceNumber": sequence_number,
        "timestamp": int(time.time() * 1000),
        "samplingRate": sampling_rate,
        "windowSize": window_size,
        "channels": channels,
    }


def build_emg_message_from_csv_row(
    row: dict,
    device_id: str,
    sequence_number: int,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    """CSV 행 데이터로 EmgWindowMessage 생성"""
    channels = []
    for ch_idx in range(DEFAULT_CHANNEL_COUNT):
        samples = []
        for s_idx in range(window_size):
            col_name = f"ch{ch_idx}_{s_idx}"
            samples.append(int(row[col_name]))
        channels.append({
            "channelIndex": ch_idx,
            "samples": samples,
        })

    return {
        "deviceId": device_id,
        "sequenceNumber": sequence_number,
        "timestamp": int(time.time() * 1000),
        "samplingRate": sampling_rate,
        "windowSize": window_size,
        "channels": channels,
    }


# ── 시연 시나리오 패턴 ──
DEMO_SCENARIO = [
    # (intent, 반복 횟수) — 각 intent를 연속으로 보내서 추론 트리거
    ("REST",  10),
    ("LEFT",  10),
    ("REST",   5),
    ("RIGHT", 10),
    ("REST",   5),
    ("STOP",   8),
    ("REST",   5),
    ("LEFT",  10),
    ("RIGHT", 10),
    ("STOP",   5),
    ("REST",   5),
]


async def run_csv_mode(
    ws_url: str,
    device_id: str,
    csv_path: str,
    interval: float,
):
    """CSV 파일의 실제 EMG 데이터를 순서대로 전송"""
    print(f"[CSV mode] Loading data from {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[CSV mode] Loaded {len(rows)} windows, connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        print(f"[CSV mode] Connected! Sending {len(rows)} windows as device={device_id}")

        for seq, row in enumerate(rows):
            msg = build_emg_message_from_csv_row(
                row=row,
                device_id=device_id,
                sequence_number=seq,
            )

            await ws.send(json.dumps(msg))
            ack = await ws.recv()
            ack_data = json.loads(ack)

            label = row.get("label", "?")
            status = "OK" if ack_data.get("success") else "FAIL"
            mode = ack_data.get("data", {}).get("mode", "?") if ack_data.get("data") else "?"
            print(
                f"  [{seq:04d}] label={label:>8s}  "
                f"mode={mode:>12s}  status={status}  "
                f"msg={ack_data.get('message', '')}"
            )

            await asyncio.sleep(interval)

    print("[CSV mode] Done")


async def run_random_mode(
    ws_url: str,
    device_id: str,
    interval: float,
    duration: Optional[float],
):
    """랜덤 intent로 무한 전송 (Ctrl+C로 중단)"""
    intents = ["REST", "LEFT", "RIGHT", "STOP"]
    print(f"[Random mode] Connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        print(f"[Random mode] Connected! Sending random EMG as device={device_id}")
        print(f"[Random mode] Press Ctrl+C to stop")

        seq = 0
        start_time = time.time()

        while True:
            if duration and (time.time() - start_time) > duration:
                break

            intent = random.choice(intents)
            msg = build_emg_message(
                device_id=device_id,
                sequence_number=seq,
                intent=intent,
            )

            await ws.send(json.dumps(msg))
            ack = await ws.recv()
            ack_data = json.loads(ack)

            status = "OK" if ack_data.get("success") else "FAIL"
            mode = ack_data.get("data", {}).get("mode", "?") if ack_data.get("data") else "?"
            print(
                f"  [{seq:04d}] intent={intent:>6s}  "
                f"mode={mode:>12s}  status={status}"
            )

            seq += 1
            await asyncio.sleep(interval)

    print("[Random mode] Done")


async def run_demo_mode(
    ws_url: str,
    device_id: str,
    interval: float,
    loop: bool = True,
):
    """시연 시나리오 모드 — REST→LEFT→RIGHT→STOP 패턴 순환"""
    print(f"[Demo mode] Connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        print(f"[Demo mode] Connected! Running demo scenario as device={device_id}")
        print(f"[Demo mode] Press Ctrl+C to stop")

        seq = 0
        cycle = 0

        while True:
            cycle += 1
            print(f"\n=== Demo cycle {cycle} ===")

            for intent, count in DEMO_SCENARIO:
                print(f"  >> Sending {count}x {intent}")

                for i in range(count):
                    msg = build_emg_message(
                        device_id=device_id,
                        sequence_number=seq,
                        intent=intent,
                    )

                    await ws.send(json.dumps(msg))
                    ack = await ws.recv()
                    ack_data = json.loads(ack)

                    status = "OK" if ack_data.get("success") else "FAIL"
                    mode = ack_data.get("data", {}).get("mode", "?") if ack_data.get("data") else "?"
                    buf = ack_data.get("data", {}).get("bufferedWindowCount", 0) if ack_data.get("data") else 0
                    print(
                        f"    [{seq:04d}] {intent:>6s}  "
                        f"mode={mode:>12s}  buf={buf:>3d}  status={status}  "
                        f"msg={ack_data.get('message', '')}"
                    )

                    seq += 1
                    await asyncio.sleep(interval)

            if not loop:
                break

    print("[Demo mode] Done")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mock EMG WebSocket Client — 시연용 EMG 데이터 전송",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "--device-id",
        default=DEFAULT_DEVICE_ID,
        help=f"Device ID (default: {DEFAULT_DEVICE_ID})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SEND_INTERVAL,
        help=f"전송 간격 초 (default: {DEFAULT_SEND_INTERVAL})",
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "random", "demo"],
        default="demo",
        help="동작 모드 (default: demo)",
    )
    parser.add_argument(
        "--csv",
        default="training/emg_features.csv",
        help="CSV 파일 경로 (csv 모드에서 사용)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="실행 시간 제한 초 (random 모드에서 사용, 미지정 시 무한)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="demo 모드에서 시나리오 1회만 실행",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  NeuroMove Mock EMG Client")
    print("=" * 60)
    print(f"  URL       : {args.url}")
    print(f"  Device ID : {args.device_id}")
    print(f"  Mode      : {args.mode}")
    print(f"  Interval  : {args.interval}s")
    print("=" * 60)
    print()
    print("  ※ 서버에서 calibration 또는 session을 먼저 시작해야 데이터가 처리됩니다")
    print("    IDLE 상태에서는 모든 데이터가 drop 됩니다")
    print()

    try:
        if args.mode == "csv":
            asyncio.run(run_csv_mode(
                ws_url=args.url,
                device_id=args.device_id,
                csv_path=args.csv,
                interval=args.interval,
            ))
        elif args.mode == "random":
            asyncio.run(run_random_mode(
                ws_url=args.url,
                device_id=args.device_id,
                interval=args.interval,
                duration=args.duration,
            ))
        else:  # demo
            asyncio.run(run_demo_mode(
                ws_url=args.url,
                device_id=args.device_id,
                interval=args.interval,
                loop=not args.no_loop,
            ))
    except KeyboardInterrupt:
        print("\n[Client] Stopped by user")
    except websockets.exceptions.ConnectionClosed:
        print("\n[Client] WebSocket connection closed")
    except ConnectionRefusedError:
        print(f"\n[Client] Connection refused — 서버가 {args.url} 에서 실행 중인지 확인하세요")


if __name__ == "__main__":
    main()
