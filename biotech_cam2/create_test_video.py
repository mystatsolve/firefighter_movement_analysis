"""테스트용 스켈레톤 동영상 생성 (사람 형태의 움직이는 스틱 피겨)"""
import cv2
import numpy as np
import os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_video.mp4')

W, H, FPS, DURATION = 640, 480, 30, 3
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT, fourcc, FPS, (W, H))

for f in range(FPS * DURATION):
    t = f / FPS
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    # 스쿼트 동작 시뮬레이션
    phase = np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz
    squat_depth = 50 * (1 - phase) / 2   # 0~50 pixel

    cx = W // 2
    # 관절 위치 계산
    head = (cx, 80)
    shoulder = (cx, 140)
    hip = (cx, int(220 + squat_depth * 0.3))
    knee = (cx, int(300 + squat_depth * 0.8))
    ankle = (cx, 400)

    l_shoulder = (cx - 40, shoulder[1])
    r_shoulder = (cx + 40, shoulder[1])
    l_elbow = (cx - 70, int(shoulder[1] + 50 + 20 * phase))
    r_elbow = (cx + 70, int(shoulder[1] + 50 + 20 * phase))
    l_wrist = (cx - 80, int(l_elbow[1] + 50))
    r_wrist = (cx + 80, int(r_elbow[1] + 50))
    l_hip = (cx - 20, hip[1])
    r_hip = (cx + 20, hip[1])
    l_knee = (cx - 25, knee[1])
    r_knee = (cx + 25, knee[1])
    l_ankle = (cx - 25, ankle[1])
    r_ankle = (cx + 25, ankle[1])

    # 그리기
    joints = [head, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist,
              l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
    connections = [
        (head, shoulder), (shoulder, l_shoulder), (shoulder, r_shoulder),
        (l_shoulder, l_elbow), (l_elbow, l_wrist),
        (r_shoulder, r_elbow), (r_elbow, r_wrist),
        (shoulder, hip), (hip, l_hip), (hip, r_hip),
        (l_hip, l_knee), (l_knee, l_ankle),
        (r_hip, r_knee), (r_knee, r_ankle),
    ]

    for p1, p2 in connections:
        cv2.line(img, p1, p2, (200, 200, 200), 3)
    for pt in joints:
        cv2.circle(img, pt, 6, (0, 255, 255), -1)

    # 머리
    cv2.circle(img, head, 20, (200, 200, 200), 2)

    cv2.putText(img, f't={t:.1f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out.write(img)

out.release()
print(f"테스트 동영상 생성: {OUTPUT}")
