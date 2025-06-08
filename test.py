# test.py

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
# 모델 성능 평가를 위한 evaluate_policy 함수 import
from stable_baselines3.common.evaluation import evaluate_policy

from env import AirSimEnv # 학습 때 사용한 환경 클래스 import

# --- 테스트 설정 ---
# 테스트할 모델 파일 경로 (원하는 체크포인트 또는 최종 모델 선택)
MODEL_PATH = "./models/a2c_airsim_model_50000_steps"
# 양적 테스트를 위해 실행할 에피소드 수
N_EVAL_EPISODES = 10

# --- 환경 및 모델 불러오기 ---
# 1. 테스트용 환경 생성
vec_env = make_vec_env(AirSimEnv, n_envs=1)

# 2. 저장된 모델 불러오기
try:
    model = A2C.load(MODEL_PATH, env=vec_env)
    print(f"모델 불러오기 성공: {MODEL_PATH}")
except Exception as e:
    print(f"모델 불러오기 실패: {e}")
    exit()

# --- 📊 1. 양적 테스트 (성능 지표 측정) ---
print("\n--- 양적 테스트 시작 ---")
# evaluate_policy 함수를 사용하여 모델 평가
mean_reward, std_reward = evaluate_policy(
    model,
    vec_env,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True # 테스트 시에는 True로 설정
)
print(f"평가 에피소드 수: {N_EVAL_EPISODES}")
print(f"평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")
print("--- 양적 테스트 종료 ---\n")


# --- 🎬 2. 질적 테스트 (실제 주행 확인) ---
print("--- 질적 테스트 시작 (주행 시뮬레이션) ---")
print("시뮬레이션 창에서 자동차의 주행을 확인하세요.")
print("종료하려면 Ctrl+C를 누르세요.")

obs = vec_env.reset()
try:
    while True:
        # deterministic=True로 설정하여 최적의 행동을 예측
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, info = vec_env.step(action)

        # 에피소드가 끝나면(충돌 등) 환경을 리셋
        if dones.any():
            print("에피소드 종료. 환경을 리셋합니다.")
            obs = vec_env.reset()

except KeyboardInterrupt:
    print("\n테스트를 종료합니다.")

finally:
    # 환경 종료
    vec_env.close()