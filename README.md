# NeuroMove AI Server

NeuroMove is a smart mobility system that interprets EMG signals to enable intuitive, hands-free movement control.  
This repository contains the AI server for the system.

EMG(근전도) 신호를 기반으로 사용자 의도를 분석하는 AI 서버입니다.

---

## 📌 프로젝트 개요

NeuroMove AI Server는 근전도(EMG) 신호를 입력으로 받아 신호 처리 및 특징 추출을 수행하고, 이를 바탕으로 사용자의 동작 의도를 추론하는 시스템입니다.

ESP32에서 전달된 EMG 데이터를 기반으로 실시간 분석을 수행하며, 최종적으로 주행 제어에 활용할 사용자 의도를 AI 서버에서 판단합니다.

---

## ⚙️ 기술 스택

- Python
- FastAPI
- Uvicorn

---

## 🚀 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/Capstone-EMG-3P1L/NeuroMove_AI.git
cd NeuroMove_AI
```

### 2. 가상환경 생성

```
python3-m venv venv
```

### 3. 가상환경 활성화

```
source venv/bin/activate
```

### 4. 패키지 설치

```
pip install-r requirements.txt
```

### 5. 서버 실행

```
uvicorn main:app--reload
```

---

## 📡 API 문서

서버 실행 후 아래 주소에서 Swagger UI를 확인할 수 있습니다.

- http://127.0.0.1:8000/docs

---

## 📂 프로젝트 구조

```
ai_server/
 ┣ main.py# FastAPI 애플리케이션 실행 진입점
 ┣ routers/# API 라우터 정의
 ┃   ┣ session_router.py
 ┃   ┣ calibration_router.py
 ┃   ┗ stream_router.py
 ┣ schemas/# 요청/응답 데이터 스키마
 ┣ services/# 신호 처리, 특징 추출, 추론 로직
 ┣ storage/# 세션 및 상태 저장소
 ┗ models/# 학습된 모델 파일 및 관련 리소스
```

---

## 📌 주요 기능

- EMG 신호 수집 데이터 처리
- 신호 전처리 (Filtering, Normalization)
- 특징 추출 (MAV, RMS 등)
- AI 기반 사용자 의도 추론

---

## 🧠 개발 예정 기능

- Calibration 기능 고도화
- 실시간 데이터 스트리밍 처리
- AI 모델 연동 및 추론 성능 개선
- WebSocket 기반 실시간 처리 확장

---

## 📝 참고 사항

- 모든 API는 FastAPI 기반으로 구성됩니다.
- 본 서버는 EMG 신호 처리 및 사용자 의도 추론을 담당합니다.
- 추후 모델 학습 및 실시간 스트리밍 기능이 추가될 예정입니다.