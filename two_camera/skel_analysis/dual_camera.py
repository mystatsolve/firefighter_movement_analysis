"""
dual_camera.py - OAK-D 듀얼 카메라 동시 관리 모듈
===================================================

OAK-D 카메라 2대(OAK-D-LITE + OAK-D)를 동시에 연결하고 관리하는 클래스를
제공합니다. test_two_cameras.py의 검증된 패턴을 기반으로 클래스화하였습니다.

주요 기능:
    - 2대 카메라 자동 감지 및 동시 연결
    - 1대만 연결된 경우 단일 카메라 모드 자동 전환
    - 비차단(non-blocking) 프레임 획득
    - 카메라 상태 모니터링 (연결 수, USB 속도)

사용 예시:
    cam = DualCameraManager()
    cam.start()
    while running:
        frames = cam.get_frames()  # [frame1, frame2] 또는 [frame1]
    cam.stop()
"""

import depthai as dai
import cv2
import numpy as np
from typing import List, Optional, Tuple

from config import OAKD_RESOLUTION_W, OAKD_RESOLUTION_H, OAKD_FPS


def _create_pipeline():
    """
    OAK-D RGB 카메라 파이프라인 생성.

    DepthAI의 파이프라인은 카메라 노드(ColorCamera)와 출력 노드(XLinkOut)를
    연결하여 프레임 스트림을 호스트(PC)로 전달하는 그래프입니다.

    Returns
    -------
    dai.Pipeline
        설정된 DepthAI 파이프라인 객체
    """
    pipeline = dai.Pipeline()

    # ColorCamera 노드: OAK-D의 RGB 센서에서 영상을 캡처
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(OAKD_RESOLUTION_W, OAKD_RESOLUTION_H)
    cam_rgb.setInterleaved(False)   # planar 포맷 (HWC가 아닌 CHW 방지)
    cam_rgb.setFps(OAKD_FPS)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # XLinkOut 노드: USB를 통해 호스트로 프레임 전송
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline


class DualCameraManager:
    """
    OAK-D 듀얼 카메라 매니저.

    2대의 OAK-D 카메라를 동시에 관리하며, 프레임을 동기적으로 제공합니다.
    1대만 연결된 경우 자동으로 단일 카메라 모드로 전환됩니다.

    Attributes
    ----------
    num_cameras : int
        현재 활성화된 카메라 수 (1 또는 2)
    is_dual : bool
        듀얼 카메라 모드 여부
    """

    def __init__(self, force_single=False):
        """
        Parameters
        ----------
        force_single : bool
            True이면 2대가 연결되어 있어도 1대만 사용 (디버깅용)
        """
        self._force_single = force_single
        self._devices: List[dai.Device] = []
        self._queues: List[dai.DataOutputQueue] = []
        self._device_names: List[str] = []
        self._running = False

    @property
    def num_cameras(self) -> int:
        """활성화된 카메라 수."""
        return len(self._devices)

    @property
    def is_dual(self) -> bool:
        """듀얼 카메라 모드 여부."""
        return len(self._devices) == 2

    def start(self) -> int:
        """
        카메라 연결 및 스트리밍 시작.

        연결된 OAK-D 디바이스를 모두 탐색하고, 최대 2대까지 파이프라인을
        생성하여 연결합니다. 1대만 발견되면 단일 모드로 자동 전환합니다.

        Returns
        -------
        int
            연결된 카메라 수 (0이면 실패)

        Raises
        ------
        RuntimeError
            카메라가 하나도 감지되지 않은 경우
        """
        # 연결된 OAK-D 디바이스 목록 조회
        device_infos = dai.Device.getAllAvailableDevices()
        print(f"[DualCamera] 감지된 OAK-D 카메라: {len(device_infos)}대")

        if len(device_infos) == 0:
            raise RuntimeError("OAK-D 카메라가 감지되지 않았습니다. USB 연결을 확인하세요.")

        # 강제 단일 모드 또는 1대만 감지된 경우
        max_cameras = 1 if self._force_single else min(len(device_infos), 2)

        pipeline = _create_pipeline()

        for i in range(max_cameras):
            info = device_infos[i]
            try:
                print(f"  카메라 {i + 1} 연결 중... (MxId: {info.mxid})")
                device = dai.Device(pipeline, dai.DeviceInfo(info.mxid))
                usb_speed = device.getUsbSpeed().name
                dev_name = device.getDeviceName()
                print(f"  카메라 {i + 1} 연결 완료 - USB: {usb_speed}, 모델: {dev_name}")

                # 프레임 출력 큐 (maxSize=4, blocking=False로 최신 프레임만 유지)
                queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

                self._devices.append(device)
                self._queues.append(queue)
                self._device_names.append(f"Cam{i + 1}({dev_name})")

            except Exception as e:
                print(f"  [WARN] 카메라 {i + 1} 연결 실패: {e}")

        if len(self._devices) == 0:
            raise RuntimeError("카메라 연결에 실패했습니다.")

        mode = "듀얼" if self.is_dual else "단일"
        print(f"\n[DualCamera] {mode} 카메라 모드 ({self.num_cameras}대 활성)")
        self._running = True
        return self.num_cameras

    def get_frames(self) -> List[Optional[np.ndarray]]:
        """
        모든 카메라에서 최신 프레임을 비차단(non-blocking) 방식으로 획득.

        tryGet()을 사용하므로 프레임이 아직 준비되지 않으면 None을 반환합니다.
        호출측에서 None 체크를 해야 합니다.

        Returns
        -------
        List[Optional[np.ndarray]]
            각 카메라의 BGR 프레임 리스트. 프레임 미준비 시 해당 원소가 None.
            예: [frame1_or_None, frame2_or_None]
        """
        frames = []
        for queue in self._queues:
            packet = queue.tryGet()
            if packet is not None:
                frames.append(packet.getCvFrame())
            else:
                frames.append(None)
        return frames

    def get_camera_info(self) -> List[str]:
        """
        연결된 카메라 이름 목록 반환.

        Returns
        -------
        List[str]
            카메라 이름 리스트 (예: ['Cam1(OAK-D-LITE)', 'Cam2(OAK-D)'])
        """
        return list(self._device_names)

    def stop(self):
        """
        모든 카메라 연결을 안전하게 종료합니다.

        각 디바이스의 close()를 호출하여 USB 연결과 내부 리소스를 해제합니다.
        이 메서드는 예외가 발생하더라도 모든 디바이스를 정리합니다.
        """
        print("[DualCamera] 카메라 종료 중...")
        for i, device in enumerate(self._devices):
            try:
                device.close()
                print(f"  카메라 {i + 1} 종료 완료")
            except Exception as e:
                print(f"  [WARN] 카메라 {i + 1} 종료 중 오류: {e}")
        self._devices.clear()
        self._queues.clear()
        self._device_names.clear()
        self._running = False
        print("[DualCamera] 모든 카메라 종료 완료")
