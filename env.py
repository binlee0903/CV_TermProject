import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2
import time
import math
import pandas as pd

class AirSimEnv(gym.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.action_space = spaces.Discrete(5)

        # --- [핵심 수정] 관측 공간을 '사전(Dictionary)' 형태로 재정의 ---
        self.observation_space = spaces.Dict({
            # 시각 정보: 84x84 크기의 3채널 이미지 (채널 0: RGB, 1: Depth, 2: Segmentation)
            "vision": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            # 상태 벡터: [경로 이탈 거리, 방향 오차, 속도]
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        self.client = airsim.CarClient()
        self.client.confirmConnection()

        try:
            self.path_df = pd.read_csv('path.csv')
            self.path_points = self.path_df[['x', 'y']].values
        except FileNotFoundError:
            print("오류: 'path.csv' 파일을 찾을 수 없습니다. 'record_path.py'를 먼저 실행하여 경로를 기록해주세요.")
            exit()

        self.car_controls = airsim.CarControls()
        self.start_time = time.time()
        self.last_collision_info = None

    def reset(self, seed=None, options=None):
        if seed is not None: super().reset(seed=seed)
        self.client.enableApiControl(False)
        self.client.reset()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.client.setCarControls(self.car_controls)
        self.start_time = time.time()
        time.sleep(0.5)

        observation, _ = self._get_observation()
        return observation, {}

    def step(self, action):
        self._do_action(action)
        time.sleep(0.05)
        self.last_collision_info = self.client.simGetCollisionInfo()
        observation, info = self._get_observation()
        reward = self._compute_reward(info)
        done = self._is_done(info)
        self.render(info)
        return observation, reward, done, False, info

    def _get_observation(self):
        requests = [
            airsim.ImageRequest("FrontCenter", airsim.ImageType.Scene, False, False),
            # DepthVis 대신 원본 깊이 데이터인 DepthPerspective 요청, 픽셀을 float으로 받음
            airsim.ImageRequest("FrontCenter", airsim.ImageType.DepthPerspective, True, False),
            airsim.ImageRequest("FrontCenter", airsim.ImageType.Segmentation, False, False)
        ]
        responses = self.client.simGetImages(requests)

        # --- 이미지 처리 부분 수정 ---

        # RGB(Scene) 이미지 처리
        img_rgb_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(responses[0].height, responses[0].width, 3)
        img_gray = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), (84, 84))

        # Depth 이미지 처리 [핵심 수정]
        # responses[1]에 이미지 데이터가 있는지 먼저 확인
        if responses[1].image_data_float:
            # float 데이터를 2D 이미지로 변환
            img_depth = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)
            # 0-255 범위의 8비트 이미지로 정규화하여 시각화 및 학습에 사용
            # 100미터 이상 거리는 모두 255(흰색)으로 처리
            img_depth_255 = np.clip(img_depth, 0, 100) / 100 * 255
            img_depth_8bit = img_depth_255.astype(np.uint8)
            img_depth_gray = cv2.resize(img_depth_8bit, (84, 84))
        else:
            # Depth 데이터를 받지 못했을 경우, 검은색 이미지로 대체
            img_depth_gray = np.zeros((84, 84), dtype=np.uint8)

        # Segmentation 이미지 처리
        if responses[2].image_data_uint8:
            img_seg_1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            img_seg = img_seg_1d.reshape(responses[2].height, responses[2].width, 3)
            img_seg_gray = cv2.resize(cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY), (84, 84))
        else:
            img_seg_gray = np.zeros((84, 84), dtype=np.uint8)

        # 3개의 이미지를 3개의 채널로 쌓아 하나의 관측값으로 만듦
        vision_obs = np.stack([img_gray, img_depth_gray, img_seg_gray], axis=-1)

        # --- 상태 벡터 계산 부분은 이전과 동일 ---
        car_state = self.client.getCarState()
        car_pos = car_state.kinematics_estimated.position
        car_p = np.array([car_pos.x_val, car_pos.y_val])
        car_yaw_rad = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)[2]

        distances = np.linalg.norm(self.path_points - car_p, axis=1)
        closest_idx = np.argmin(distances)
        path_deviation = distances[closest_idx]

        p1 = self.path_points[max(0, closest_idx - 1)]
        p2 = self.path_points[min(len(self.path_points) - 1, closest_idx + 1)]
        path_vector = p2 - p1
        car_vector = car_p - p1
        if np.cross(path_vector, car_vector) > 0:
            path_deviation = -path_deviation

        lookahead_idx = min(closest_idx + 10, len(self.path_points) - 1)
        target_yaw_rad = math.atan2(self.path_points[lookahead_idx][1] - car_p[1], self.path_points[lookahead_idx][0] - car_p[0])
        heading_error = target_yaw_rad - car_yaw_rad
        if heading_error > math.pi: heading_error -= 2 * math.pi
        if heading_error < -math.pi: heading_error += 2 * math.pi

        state_obs = np.array([path_deviation, heading_error, car_state.speed], dtype=np.float32)

        observation = {"vision": vision_obs, "state": state_obs}
        info = {'deviation': path_deviation, 'heading_error': heading_error}

        return observation, info

    def _compute_reward(self, info):
        if self.last_collision_info.has_collided:
            return -20.0

        # 경로 유지에 대한 보상 강화
        deviation_reward = math.exp(-0.5 * abs(info['deviation'])) # 더 뾰족하게 만들어 중앙 유지 유도
        heading_reward = math.exp(-1.0 * abs(info['heading_error']))

        # 기본 보상은 경로 유지 점수
        reward = (deviation_reward + heading_reward) / 2

        return reward

    def _is_done(self, info):
        if self.last_collision_info.has_collided:
            print("Collision detected.")
            return True
        if abs(info['deviation']) > 8.0:
            print(f"Too far from the path: {info['deviation']:.2f}m")
            return True
        return False

    def _do_action(self, action):
        self.car_controls.throttle = 0.4
        self.car_controls.brake = 0
        if action == 0: self.car_controls.steering = 0
        elif action == 1: self.car_controls.steering = -0.25
        elif action == 2: self.car_controls.steering = 0.25
        elif action == 3: self.car_controls.steering = -0.7
        elif action == 4: self.car_controls.steering = 0.7
        self.client.setCarControls(self.car_controls)

    def render(self, info):
        print(f"Path Dev: {info.get('deviation', 0):+5.2f}m | Head Err: {math.degrees(info.get('heading_error', 0)):+5.1f}deg", end='\r')

    def close(self):
        self.client.enableApiControl(False)
        print("\nTraining finished.")