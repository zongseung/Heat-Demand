GLAM: GAM–LSTM Additive Model
=============================

이 레포는 논문 "GLAM: A Hybrid GAM with LSTM Framework for District Heating Production Forecasting"
의 아키텍처를 기반으로 한 구현/실험을 포함합니다. 핵심 아이디어는
**구조적(해석 가능한) 성분**과 **잔차 동역학**을 분리해 예측 성능과 설명력을 동시에 확보하는 것입니다.

아키텍처 개요
-------------

GLAM은 2‑Stage 구조입니다.

Stage I: 구조적 베이스라인 (GAM)
--------------------------------

목표:
  - 장기 추세, 날씨 비선형 반응, 다중 계절성, 사회적 주기(휴일/주말)를
    해석 가능한 형태로 분리해 구조 성분 L_hat을 추정.

구성:
  1) 추세 + 변경점 (changepoint)
     - 선형 추세 + piecewise changepoint: (t - c_j)+
     - 논문에서는 L1 패널티로 sparse changepoint를 유도.
  2) 날씨 반응 (HDH)
     - HDH = max(0, T_base - T)
     - HDH에 대한 비선형 스무딩 (B‑spline)
  3) 다중 계절성
     - 연/일 주기: Fourier series (sin/cos)
  4) 사회적 주기
     - 주말/공휴일 indicator

구현 방식:
  - GAM으로 trend + weather + seasonal + social 성분을 학습.
  - changepoint는 L1(Lasso)로 따로 추정해 GAM 출력에 더함.
    (pyGAM이 L1을 직접 지원하지 않기 때문)

Stage II: 잔차 동역학 (Seq2Seq LSTM)
-----------------------------------

목표:
  - Stage I 잔차 r_t = y_t - L_hat_t 에 남은 관성/단기 의존성 학습.
  - 다시간(H‑step) 예측을 위해 Seq2Seq LSTM 사용.

구성:
  - 입력: 과거 L 길이의 잔차 시퀀스
  - 출력: H 길이의 잔차 보정치 D_hat
  - 최종 예측: y_hat = L_hat + D_hat

데이터 처리/피처
---------------

필수 컬럼:
  - date, heat_demand, imputated_temperature

피처:
  - t (시간 인덱스)
  - HDH (Heating Degree Hours)
  - Fourier terms (daily/yearly)
  - is_non_working (주말/공휴일)
  - changepoint hinge features (cp_j)

학습/평가 프로토콜
------------------

논문 프로토콜:
  - Rolling-origin 방식 권장 (각 cutoff마다 Stage I/II 재학습).

현재 노트북 기준:
  - Sliding-window 방식으로 구현 가능.
  - Sliding-window = 고정 길이 윈도우만 학습 (누적 학습 아님).
  - non‑overlap 여부는 STRIDE 값으로 결정:
      * STRIDE == HORIZON  -> non‑overlap
      * STRIDE == 1        -> overlap

하이퍼파라미터 탐색
-------------------

Stage I:
  - Fourier 차수 K_daily, K_yearly
  - changepoint 개수 K_cp
  - n_splines (HDH 스무딩)
  - smoothing lambda (pyGAM gridsearch)

Stage II:
  - hidden_size, num_layers, dropout, lr, batch_size 등
  - teacher forcing 비율

파일 구성
---------

주요 파일:
  - GLAM_implementation.ipynb
    * 논문 구조 기반 Stage I/II 구현
    * 구조/잔차 분리, GAM + LSTM 통합 예측
  - #5_대회준비_모델링.Ipynb
    * 대회용 베이스라인/실험 기록 (논문 구조와는 차이 있음)
  - gangnam_CHP.csv
    * 입력 데이터
  - GLAM (1).pdf
    * 논문 원문

실행 흐름 (권장 순서)
---------------------

1) 데이터 로드 및 HDH/Calendar feature 생성
2) Fourier/Changepoint 구조 탐색
3) Stage I (GAM) 학습 + Lasso changepoint
4) Stage I 잔차 계산
5) Stage II (Seq2Seq LSTM) 학습
6) 통합 예측 (L_hat + D_hat)
7) Sliding/Rolling 평가

비고
----
- changepoint는 논문에서 L1 패널티를 사용하므로 Lasso 방식으로 분리 구현.
- pyGAM의 smoothing lambda는 gridsearch로 튜닝 가능.
