# train.py

import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
# CheckpointCallback을 import 합니다.
from stable_baselines3.common.callbacks import CheckpointCallback

from env import AirSimEnv # 이전에 정의한 환경 클래스 import

# --- 1. 모델과 로그가 저장될 경로 설정 ---
save_dir = "./models/"
log_dir = "./a2c_airsim_tensorboard/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# Vectorized 환경 생성
vec_env = make_vec_env(AirSimEnv, n_envs=1)


# --- 2. 10000 스텝마다 모델을 저장하는 콜백 설정 ---
# save_freq: 저장 빈도 (스텝 단위)
# save_path: 저장될 폴더 경로
# name_prefix: 저장될 모델 파일의 접두사
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=save_dir,
    name_prefix="a2c_airsim_model"
)


# A2C 모델 정의
# 여러 입력을 받으므로 'MultiInputPolicy' 사용
model = A2C(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir
)


# --- 3. learn() 함수에 callback 인자 추가 ---
# total_timesteps는 충분히 큰 값으로 설정
# 학습을 진행하면서 10000 스텝마다 checkpoint_callback이 호출됩니다.
model.learn(
    total_timesteps=50000,
    callback=checkpoint_callback # 콜백을 전달
)


# 최종 모델 저장 (학습이 모두 완료된 후)
model.save(f"{save_dir}/a2c_airsim_driver_final")


# 학습 종료 후 환경 닫기
vec_env.close()