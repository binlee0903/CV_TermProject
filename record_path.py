import airsim
import time
import csv
import os

# 파일 경로 설정
path_file = "path.csv"

# 파일이 이미 존재하면 삭제하여 새로 시작
if os.path.exists(path_file):
    os.remove(path_file)
    print(f"'{path_file}'이 존재하여 삭제하고 새로 시작합니다.")

# AirSim 클라이언트 연결
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False) # 수동 운전을 위해 API 제어 비활성화

print("수동으로 주행을 시작하세요. 3초 후부터 경로 기록을 시작합니다.")
print("기록을 마치려면 Ctrl+C를 눌러 프로그램을 종료하세요.")
time.sleep(3)

try:
    with open(path_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y']) # 헤더 작성

        print("경로 기록 시작!")
        while True:
            # 자동차 상태 가져오기
            car_state = client.getCarState()
            pos = car_state.kinematics_estimated.position

            # X, Y 좌표만 CSV 파일에 기록
            writer.writerow([pos.x_val, pos.y_val])
            print(f"기록됨: X={pos.x_val:.2f}, Y={pos.y_val:.2f}")

            # 0.2초 간격으로 위치 기록
            time.sleep(0.2)

except KeyboardInterrupt:
    client.reset()
    print(f"\n경로 기록이 중단되고 '{path_file}'에 저장되었습니다.")