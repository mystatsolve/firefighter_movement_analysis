"""
OAK-D 카메라 2대 동시 테스트
- 두 카메라의 RGB 영상을 동시에 표시
- ESC 또는 'q' 키로 종료
"""

import depthai as dai
import cv2
import time


def create_pipeline():
    """OAK-D RGB 카메라 파이프라인 생성"""
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline


def main():
    print("=== OAK-D 카메라 2대 동시 테스트 ===\n")

    # 연결된 디바이스 검색
    device_infos = dai.Device.getAllAvailableDevices()
    print(f"감지된 카메라 수: {len(device_infos)}")

    if len(device_infos) < 2:
        print("오류: 2대의 카메라가 필요합니다!")
        print("현재 감지된 카메라:")
        for i, info in enumerate(device_infos):
            print(f"  {i}: MxId={info.mxid}, State={info.state.name}")
        return

    for i, info in enumerate(device_infos):
        print(f"  카메라 {i}: MxId={info.mxid}")

    # 두 카메라 동시 연결
    pipeline = create_pipeline()

    print("\n카메라 1 연결 중...")
    dev1 = dai.Device(pipeline, dai.DeviceInfo(device_infos[0].mxid))
    print(f"  카메라 1 연결 완료 - USB: {dev1.getUsbSpeed().name}, 이름: {dev1.getDeviceName()}")

    print("카메라 2 연결 중...")
    dev2 = dai.Device(pipeline, dai.DeviceInfo(device_infos[1].mxid))
    print(f"  카메라 2 연결 완료 - USB: {dev2.getUsbSpeed().name}, 이름: {dev2.getDeviceName()}")

    q1 = dev1.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q2 = dev2.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    print("\n두 카메라 영상 표시 중... (ESC 또는 'q'로 종료)\n")

    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            in1 = q1.tryGet()
            in2 = q2.tryGet()

            if in1 is not None:
                frame1 = in1.getCvFrame()
                cv2.putText(frame1, "Camera 1", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera 1", frame1)

            if in2 is not None:
                frame2 = in2.getCvFrame()
                cv2.putText(frame2, "Camera 2", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Camera 2", frame2)

            # FPS 계산
            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = time.time() - fps_time
                fps = frame_count / elapsed
                print(f"  FPS: {fps:.1f}")
                frame_count = 0
                fps_time = time.time()

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        print("\n카메라 종료 중...")
        dev1.close()
        dev2.close()
        cv2.destroyAllWindows()
        print("완료!")


if __name__ == "__main__":
    main()
