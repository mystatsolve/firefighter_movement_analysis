"""
OAK-D Camera Pipeline - DepthAI RGB stream with optional stereo depth.
Falls back to webcam when OAK-D is not available or --webcam is specified.

=============================================================================
모듈 목적 및 아키텍처 (Module Purpose & Architecture)
=============================================================================

이 모듈은 Luxonis OAK-D 카메라를 DepthAI SDK를 통해 제어하는 파이프라인을 구현합니다.
OAK-D는 Intel Movidius Myriad X VPU(Vision Processing Unit)를 내장한 스테레오 깊이
카메라로, 온디바이스(on-device) 신경망 추론과 스테레오 깊이 계산이 가능합니다.

OAK-D 카메라 하드웨어 구성:
    - RGB 카메라: Sony IMX378 (12MP), 최대 4K 해상도, 중앙에 위치
    - 스테레오 모노 카메라: 2x OmniVision OV9282 (1MP), 좌/우 배치
      글로벌 셔터 방식으로 움직이는 물체 촬영 시 왜곡이 적음
    - 온보드 프로세서: Intel Movidius Myriad X VPU
      신경망 추론, 이미지 처리, 스테레오 매칭을 하드웨어 가속으로 수행
    - 인터페이스: USB3 (USB2도 호환되나 대역폭 제한 있음)

DepthAI 파이프라인 개념:
    - DepthAI는 "노드 그래프(Node Graph)" 기반 아키텍처를 사용합니다.
    - 각 노드(Node)는 카메라 센서, 이미지 처리, 신경망 추론, 데이터 출력 등의
      특정 기능을 담당합니다.
    - 노드 간 연결(Link)을 통해 데이터가 흐르며, 이 전체 구조를 "파이프라인"이라 합니다.
    - 파이프라인은 OAK-D 디바이스의 VPU에서 실행되며, 호스트 PC와는 USB XLink
      프로토콜을 통해 데이터를 주고받습니다.
    - 파이프라인 정의(Python) -> 직렬화 -> USB 전송 -> VPU에서 실행
      이 과정은 dai.Device(pipeline) 호출 시 자동으로 수행됩니다.

USB XLink 프로토콜 동작 원리:
    - XLink는 Luxonis가 개발한 USB 통신 프로토콜입니다.
    - OAK-D 디바이스(VPU)와 호스트 PC 간의 양방향 데이터 스트림을 관리합니다.
    - XLinkOut 노드: 디바이스 -> 호스트 방향으로 데이터를 전송합니다.
      (카메라 프레임, 깊이맵, 신경망 추론 결과 등)
    - XLinkIn 노드: 호스트 -> 디바이스 방향으로 데이터를 전송합니다.
      (설정 변경, 이미지 전송 등)
    - 각 스트림은 고유한 이름(streamName)으로 식별됩니다.
    - 호스트에서는 getOutputQueue()로 수신 큐를 생성하여 프레임을 받아옵니다.
    - maxSize와 blocking 파라미터로 큐 오버플로우 시 동작을 제어할 수 있습니다:
      - blocking=False: 큐가 가득 차면 가장 오래된 프레임을 버립니다 (실시간 처리에 적합)
      - blocking=True: 큐가 가득 차면 새 프레임이 들어올 때까지 대기합니다 (녹화에 적합)

본 모듈의 파이프라인 노드 그래프 구조:

    [RGB 스트림 - 항상 활성화]
      ColorCamera (cam_rgb)
           |
           | preview 출력 (setPreviewSize로 지정한 크기)
           v
      XLinkOut ("rgb")  --USB XLink-->  호스트 PC (rgb_queue)

    [깊이 스트림 - enable_depth=True일 때만 활성화]
      MonoCamera (left)     MonoCamera (right)
           |                      |
           | out                  | out
           v                      v
           +----------------------+
           |   StereoDepth        |
           |   (스테레오 매칭,     |
           |    시차->깊이 변환)   |
           +----------+-----------+
                      | depth 출력 (uint16, mm 단위)
                      v
      XLinkOut ("depth")  --USB XLink-->  호스트 PC (depth_queue)

웹캠 폴백(Fallback) 메커니즘:
    - OAK-D가 연결되지 않았거나 초기화에 실패한 경우, 자동으로 일반 USB 웹캠으로
      전환됩니다. (ImportError, RuntimeError 등 모든 예외를 catch)
    - 사용자가 명시적으로 --webcam 옵션을 지정한 경우에도 웹캠 모드로 동작합니다.
    - 웹캠 모드에서는 깊이(depth) 데이터를 사용할 수 없으며, depth는 항상 None입니다.
    - OpenCV의 cv2.VideoCapture를 사용하여 프레임을 캡처합니다.
    - MediaPipe는 단안(monocular) RGB 이미지만으로도 3D 포즈를 추정할 수 있으므로,
      웹캠 모드에서도 포즈 분석 기능은 정상적으로 동작합니다.

사용 예시:
    camera = OAKDCamera(use_webcam=False, enable_depth=True)
    camera.start()

    while True:
        bgr, depth = camera.get_frame()
        if bgr is not None:
            # bgr: numpy.ndarray (H, W, 3), dtype=uint8, BGR 색상 순서
            # depth: numpy.ndarray (H, W), dtype=uint16, 밀리미터 단위 또는 None
            pass

    camera.stop()
"""

# =============================================================================
# 표준 라이브러리 및 서드파티 패키지 임포트
# =============================================================================

# OpenCV: 컴퓨터 비전 라이브러리. 웹캠 캡처, 이미지 처리, 화면 표시에 사용됩니다.
# 웹캠 폴백 모드에서 cv2.VideoCapture로 프레임을 획득합니다.
import cv2

# NumPy: 다차원 배열 연산 라이브러리. 이미지 데이터는 numpy 배열로 표현됩니다.
# 이 모듈에서 직접 사용하지는 않지만, 반환되는 프레임이 numpy.ndarray 타입입니다.
import numpy as np

# config 모듈에서 카메라 관련 설정값들을 가져옵니다.
# 이 값들은 config.py에서 중앙 관리되며, 프로젝트 전체에서 일관된 설정을 보장합니다.
#
# OAKD_RESOLUTION_W : int - 카메라 출력 프레임의 가로 픽셀 수 (예: 640)
# OAKD_RESOLUTION_H : int - 카메라 출력 프레임의 세로 픽셀 수 (예: 480)
# OAKD_FPS          : int - 초당 프레임 수 (예: 30)
# OAKD_PREVIEW_SIZE : tuple - 미리보기 크기 (네트워크 입력용, 이 모듈에서는 미사용)
# OAKD_DEPTH_ALIGN_TO_RGB : bool - 깊이맵을 RGB 카메라 좌표계에 정렬할지 여부
from config import (
    OAKD_RESOLUTION_W, OAKD_RESOLUTION_H, OAKD_FPS,
    OAKD_PREVIEW_SIZE, OAKD_DEPTH_ALIGN_TO_RGB,
)


class OAKDCamera:
    """
    OAK-D DepthAI 파이프라인 기반 RGB (+ 선택적 깊이맵) 스트리밍 클래스.

    이 클래스는 카메라 하드웨어와의 모든 상호작용을 캡슐화합니다.
    메인 루프는 get_frame()만 호출하면 되며, 내부적으로 OAK-D인지
    웹캠인지에 따라 적절한 프레임 획득 방법을 사용합니다 (Strategy 패턴).

    클래스의 생명주기(Lifecycle):
        1. __init__() : 설정값 저장 (하드웨어 접근 없음)
        2. start()    : 카메라 초기화 및 스트리밍 시작
        3. get_frame() : 프레임 획득 (반복 호출)
        4. stop()     : 자원 해제 및 연결 종료

    인터페이스:
        start()      -> 카메라 초기화 및 스트리밍 시작
        get_frame()  -> (BGR 프레임, 깊이맵 또는 None) 반환
        stop()       -> 자원 해제 및 연결 종료
        is_webcam    -> 현재 웹캠 모드 여부 (property)
        is_started   -> 스트리밍 시작 여부 (property)

    Attributes
    ----------
    _use_webcam : bool
        True이면 웹캠 모드, False이면 OAK-D 모드.
        start() 중 OAK-D 초기화 실패 시 자동으로 True로 변경될 수 있음.
    _enable_depth : bool
        스테레오 깊이맵 활성화 여부.
        OAK-D 모드에서만 유효하며, 웹캠 모드에서는 무시됨.
    _webcam_id : int
        웹캠 장치 번호. cv2.VideoCapture에 전달되는 디바이스 인덱스.
        0 = 첫 번째 카메라 (보통 내장 웹캠), 1 = 두 번째 카메라.
    _pipeline : dai.Pipeline 또는 None
        DepthAI 파이프라인 객체. 노드 그래프를 정의하는 컨테이너.
        OAK-D 모드에서만 사용되며, 웹캠 모드에서는 None.
    _device : dai.Device 또는 None
        OAK-D 하드웨어 디바이스 핸들. USB 연결을 관리.
        OAK-D 모드에서만 사용되며, 웹캠 모드에서는 None.
    _rgb_queue : dai.DataOutputQueue 또는 None
        RGB 프레임 수신 큐. XLink를 통해 호스트로 전달되는 프레임을 버퍼링.
    _depth_queue : dai.DataOutputQueue 또는 None
        깊이맵 수신 큐. 스테레오 깊이 활성화 시에만 생성됨.
    _cap : cv2.VideoCapture 또는 None
        웹캠 폴백 시 사용하는 OpenCV 비디오 캡처 객체.
    _started : bool
        카메라 스트리밍이 성공적으로 시작되었는지 여부.
    """

    def __init__(self, use_webcam=False, enable_depth=False, webcam_id=0):
        """
        OAKDCamera 인스턴스를 초기화합니다.

        이 단계에서는 실제로 카메라 하드웨어에 접근하지 않습니다.
        설정값만 저장하고, start() 메서드 호출 시 실제 하드웨어 초기화가 이루어집니다.
        이렇게 생성과 초기화를 분리하는 이유:
        - 초기화 실패 시 적절한 에러 처리(폴백)를 가능하게 하기 위함
        - 객체 생성 시점과 카메라 시작 시점을 분리하여 유연한 제어 가능
        - 테스트 시 하드웨어 없이도 객체 생성 가능

        Parameters
        ----------
        use_webcam : bool, default=False
            True로 설정하면 OAK-D 대신 일반 USB 웹캠을 사용합니다.
            커맨드라인에서 --webcam 플래그에 대응됩니다.
            개발/디버깅 시 OAK-D 없이도 파이프라인을 테스트할 수 있습니다.
            False로 설정해도 OAK-D 연결 실패 시 자동으로 True로 변경됩니다.

        enable_depth : bool, default=False
            True로 설정하면 스테레오 깊이(Stereo Depth) 파이프라인을 활성화합니다.
            좌/우 모노 카메라로부터 시차(disparity)를 계산하여 깊이맵을 생성합니다.
            깊이 데이터가 필요 없는 경우 False로 두면 USB 대역폭과 VPU 처리 성능이
            절약됩니다. 웹캠 모드에서는 이 옵션이 무시됩니다 (스테레오 카메라 없음).

        webcam_id : int, default=0
            웹캠 폴백 시 사용할 카메라 디바이스 인덱스입니다.
            시스템에 여러 카메라가 연결된 경우 0, 1, 2... 순서로 식별됩니다.
            일반적으로 노트북 내장 웹캠이 0번, 외부 USB 웹캠이 1번부터 시작합니다.
            Windows에서는 DirectShow를 통해 열거된 순서를 따릅니다.
        """
        # --- 사용자 설정값 저장 ---
        self._use_webcam = use_webcam        # 웹캠 모드 사용 여부
        self._enable_depth = enable_depth    # 깊이 스트림 활성화 여부
        self._webcam_id = webcam_id          # 웹캠 디바이스 인덱스 번호

        # --- DepthAI 관련 내부 상태 변수 (OAK-D 모드에서만 사용) ---
        self._pipeline = None       # dai.Pipeline: 노드 그래프 정의 객체
        self._device = None         # dai.Device: OAK-D 하드웨어 연결 핸들
        self._rgb_queue = None      # dai.DataOutputQueue: RGB 프레임 수신 큐
        self._depth_queue = None    # dai.DataOutputQueue: 깊이맵 수신 큐 (선택적)

        # --- 웹캠 폴백 관련 내부 상태 변수 ---
        self._cap = None  # cv2.VideoCapture: 웹캠 폴백용 OpenCV 캡처 객체

        # --- 공통 상태 플래그 ---
        self._started = False       # 스트리밍 시작 여부 플래그

    # =========================================================================
    # 카메라 시작/정지 메서드
    # =========================================================================

    def start(self):
        """
        카메라를 초기화하고 프레임 스트리밍을 시작합니다.

        동작 흐름:
            1. use_webcam=True인 경우:
               -> 즉시 _start_webcam() 호출하여 웹캠 모드로 시작

            2. use_webcam=False인 경우:
               a. _start_oakd() 호출하여 OAK-D 초기화 시도
               b. 성공 시 -> OAK-D 모드로 동작
               c. 실패 시 -> 경고 메시지 출력 후 자동으로 웹캠 모드로 폴백

        폴백 메커니즘이 자동 처리하는 상황들:
            - OAK-D가 물리적으로 연결되지 않은 경우
            - DepthAI 드라이버(depthai 패키지)가 설치되지 않은 경우 (ImportError)
            - USB 연결 오류 또는 디바이스 펌웨어 문제 (RuntimeError)
            - 다른 프로세스가 이미 OAK-D를 점유하고 있는 경우
            - USB 대역폭 부족 (USB2 허브에 여러 장치 연결 시)

        Raises
        ------
        RuntimeError
            웹캠 모드에서도 카메라를 열 수 없는 경우 발생합니다.
            (예: 시스템에 어떤 카메라도 연결되지 않은 경우)
        """
        if self._use_webcam:
            # 사용자가 명시적으로 웹캠 모드를 선택한 경우 -> 바로 웹캠 시작
            self._start_webcam()
        else:
            # OAK-D 모드 시도 -> 실패 시 자동으로 웹캠으로 폴백
            try:
                self._start_oakd()
            except Exception as e:
                # OAK-D 초기화 실패 원인을 콘솔에 출력 (디버깅 용도)
                # ImportError: depthai 패키지 미설치
                # RuntimeError: OAK-D 미연결 또는 USB 오류
                # 기타: 펌웨어 오류, 권한 문제 등
                print(f"[WARN] OAK-D init failed: {e}")
                print("[INFO] Falling back to webcam...")
                # 내부 상태를 웹캠 모드로 전환하고 웹캠 시작
                self._use_webcam = True
                self._start_webcam()
        # 카메라 종류에 관계없이 시작 완료 플래그 설정
        self._started = True

    def _start_oakd(self):
        """
        OAK-D DepthAI 파이프라인을 구성하고 디바이스를 시작합니다.

        이 메서드는 DepthAI 파이프라인의 전체 구성 과정을 수행합니다:

        ═══════════════════════════════════════════════════════════════════
        1단계: Pipeline 객체 생성
        ═══════════════════════════════════════════════════════════════════
        - Pipeline은 노드들의 컨테이너이자 실행 그래프의 정의입니다.
        - Python에서 정의한 파이프라인은 직렬화되어 OAK-D VPU로 전송됩니다.
        - 아직 실행되지 않으며, dai.Device()에 전달될 때 VPU에 업로드됩니다.

        ═══════════════════════════════════════════════════════════════════
        2단계: ColorCamera 노드 설정
        ═══════════════════════════════════════════════════════════════════
        - OAK-D의 중앙 RGB 카메라 (Sony IMX378, 12MP)를 제어합니다.
        - setPreviewSize(): 미리보기 출력 크기 설정 (VPU 하드웨어 스케일링)
        - setInterleaved(False): 픽셀 데이터를 평면(planar/CHW) 형식으로 출력
          * Interleaved (HWC): [B0,G0,R0, B1,G1,R1, ...] - 픽셀별 교차 배치
          * Planar (CHW): [B0,B1,..., G0,G1,..., R0,R1,...] - 채널별 분리 배치
          * getCvFrame() 호출 시 자동으로 OpenCV 호환 HWC 형식으로 변환됨
        - setColorOrder(BGR): OpenCV 호환 BGR 색상 순서
        - setFps(): 센서의 프레임 레이트 제한
        - setResolution(THE_1080_P): 센서 캡처 해상도 (ISP에서 처리)
          * 센서는 1080P로 캡처한 후, preview 크기로 하드웨어 스케일다운
          * 고해상도 센서의 화질을 유지하면서 출력 크기는 줄일 수 있음

        ═══════════════════════════════════════════════════════════════════
        3단계: XLinkOut 노드 설정 (RGB 스트림)
        ═══════════════════════════════════════════════════════════════════
        - 디바이스에서 호스트로 데이터를 전송하는 출력 노드입니다.
        - setStreamName("rgb"): 호스트에서 이 스트림을 식별할 고유 이름
        - cam_rgb.preview.link(xout_rgb.input): 카메라 미리보기 -> XLink 출력 연결
        - link()는 단방향 데이터 흐름을 정의: 카메라 -> USB -> 호스트

        ═══════════════════════════════════════════════════════════════════
        4단계: 스테레오 깊이 파이프라인 (선택적)
        ═══════════════════════════════════════════════════════════════════
        - MonoCamera (좌/우): OAK-D 양쪽의 흑백 카메라 (OV9282)
          * THE_400_P = 640x400 해상도 (깊이 계산에 충분, 성능 효율적)
          * 글로벌 셔터로 움직임 왜곡 최소화
        - StereoDepth: 좌/우 이미지로부터 시차(disparity) 계산
          * 시차 = 같은 물체가 좌/우 이미지에서 수평으로 이동한 픽셀 수
          * 깊이(distance) = baseline * focal_length / disparity
          * baseline: 좌/우 카메라 간 물리적 거리 (OAK-D: 약 7.5cm)
          * HIGH_DENSITY 프리셋: 서브픽셀 디스패리티 + 확장 디스패리티 활성화
          * HIGH_ACCURACY 프리셋: 높은 신뢰도 영역만 깊이 추정
        - setDepthAlign(CAM_A): 깊이맵을 RGB 카메라 시점으로 정렬
          * RGB 이미지의 각 픽셀(x,y)에 대응하는 깊이값 직접 사용 가능

        ═══════════════════════════════════════════════════════════════════
        5단계: Device 생성 및 파이프라인 업로드
        ═══════════════════════════════════════════════════════════════════
        - dai.Device(pipeline): OAK-D에 USB 연결, 파이프라인을 VPU에 업로드
        - 이 시점에서 카메라 스트리밍이 즉시 시작됩니다.
        - 연결 실패 시 예외 발생 -> 상위 start()에서 웹캠 폴백 처리

        ═══════════════════════════════════════════════════════════════════
        6단계: 출력 큐 생성
        ═══════════════════════════════════════════════════════════════════
        - getOutputQueue(): XLink 수신 큐 생성
        - maxSize=4: 최대 4프레임 버퍼링
          * 작을수록 지연 감소, 클수록 프레임 드롭 감소
        - blocking=False: 큐 가득 참 -> 오래된 프레임 자동 폐기 (실시간용)
          * blocking=True: 큐 가득 참 -> 디바이스 측 대기 (녹화용)

        Note
        ----
        depthai 패키지는 이 메서드 내부에서만 import합니다 (지연 임포트).
        이렇게 하면 OAK-D가 없는 환경(depthai 미설치)에서도
        웹캠 모드로 이 모듈을 사용할 수 있습니다.

        Raises
        ------
        ImportError
            depthai 패키지가 설치되지 않은 경우 발생합니다.
        RuntimeError
            OAK-D 디바이스가 연결되지 않았거나 USB 오류 발생 시 발생합니다.
        """
        # depthai 패키지를 지연 임포트(lazy import)합니다.
        # 이유: OAK-D를 사용하지 않는 환경에서도 이 모듈을 import할 수 있게 하기 위함.
        # depthai가 설치되지 않은 경우 ImportError가 발생하여 상위 start()에서
        # except로 잡히고, 웹캠 폴백이 실행됩니다.
        import depthai as dai

        # =====================================================================
        # 파이프라인 정의 시작
        # =====================================================================
        # Pipeline 객체는 DepthAI의 핵심 컨테이너로, 모든 노드와 연결을 포함합니다.
        # Python 측에서 그래프를 정의한 후, dai.Device()에 전달하면
        # 직렬화되어 OAK-D의 Myriad X VPU로 업로드됩니다.
        # VPU에서는 이 그래프에 따라 데이터를 처리하고 결과를 USB로 전송합니다.
        pipeline = dai.Pipeline()

        # =====================================================================
        # RGB 컬러 카메라 노드 생성 및 설정
        # =====================================================================
        # ColorCamera 노드: OAK-D의 중앙 RGB 카메라 센서(Sony IMX378)를 제어합니다.
        # 이 노드는 여러 출력 포트를 가집니다:
        #   - preview: 지정한 크기로 리사이즈된 프레임 (신경망 입력에 적합)
        #   - video: 원본 해상도 또는 비디오 인코더용 출력
        #   - still: 고해상도 정지 이미지 캡처용 출력 (트리거 방식)
        #   - isp: ISP(Image Signal Processor) 처리 후 원본 크기 출력
        # 본 프로젝트에서는 preview 출력을 사용합니다.
        cam_rgb = pipeline.create(dai.node.ColorCamera)

        # 미리보기(preview) 출력 크기를 설정합니다.
        # VPU의 하드웨어 스케일러가 센서 해상도(1080P)에서 이 크기로 축소합니다.
        # CPU 부하 없이 리사이즈되므로, 호스트에서 별도 리사이즈가 불필요합니다.
        # 포즈 추정 모델의 입력 크기에 맞추면 추가 전처리를 줄일 수 있습니다.
        cam_rgb.setPreviewSize(OAKD_RESOLUTION_W, OAKD_RESOLUTION_H)

        # 인터리브(Interleaved) 모드를 비활성화합니다.
        # - Interleaved = True  (HWC 형식): [B0,G0,R0, B1,G1,R1, ...] 픽셀별 교차 배치
        # - Interleaved = False (CHW 형식): [B0,B1,..., G0,G1,..., R0,R1,...] 채널별 분리
        # False로 설정하면 VPU 내부에서 CHW(planar) 형식으로 출력하지만,
        # 호스트에서 getCvFrame() 호출 시 자동으로 HWC(OpenCV 호환)로 변환됩니다.
        # 따라서 실질적으로 OpenCV에서 바로 사용 가능한 형태가 됩니다.
        cam_rgb.setInterleaved(False)

        # 색상 순서를 BGR로 설정합니다.
        # OpenCV는 역사적 이유로 RGB가 아닌 BGR 순서를 기본으로 사용합니다.
        # BGR로 설정하면 cv2.imshow(), cv2.imwrite(), cv2.cvtColor() 등과
        # 호환되어 별도의 색상 변환(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 불필요.
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # 카메라 프레임 레이트(FPS)를 설정합니다.
        # 높은 FPS: 부드러운 영상, USB 대역폭/처리 부하 증가
        # 낮은 FPS: 끊김 느낌, 대역폭/부하 감소
        # 포즈 추정에는 일반적으로 15~30 FPS가 적합합니다.
        # 이 값은 센서 측 FPS 제한이며, 호스트 처리 속도가 느리면 프레임이 큐에 쌓임.
        cam_rgb.setFps(OAKD_FPS)

        # 센서 캡처 해상도를 1080P (1920x1080, Full HD)로 설정합니다.
        # 센서는 이 해상도로 캡처한 후, ISP에서 화이트밸런스/노이즈 감소 등을 처리하고,
        # preview 출력에서는 setPreviewSize()로 지정한 크기로 스케일다운합니다.
        # 사용 가능한 센서 해상도 옵션:
        #   THE_1080_P  = 1920x1080 (Full HD) - 가장 범용적
        #   THE_4_K     = 3840x2160 (4K UHD) - 고해상도 필요 시
        #   THE_12_MP   = 4056x3040 (최대 해상도) - 정지 이미지용
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # =====================================================================
        # XLinkOut 노드 생성 - RGB 프레임을 호스트 PC로 전송
        # =====================================================================
        # XLinkOut은 디바이스(OAK-D VPU)에서 호스트(PC)로 데이터를 전송하는 노드입니다.
        # USB 케이블을 통해 XLink 프로토콜로 프레임 데이터가 바이너리 직렬화되어 전달됩니다.
        # 하나의 파이프라인에 여러 XLinkOut 노드를 생성할 수 있으며,
        # 각 노드는 고유한 스트림 이름(streamName)으로 식별됩니다.
        xout_rgb = pipeline.create(dai.node.XLinkOut)

        # 스트림 이름 설정: 호스트에서 getOutputQueue(name="rgb")로 이 스트림에 접근합니다.
        # 이 이름은 파이프라인 내에서 유일해야 하며, 호스트 큐 생성 시 동일한 이름을 사용.
        xout_rgb.setStreamName("rgb")

        # ColorCamera의 preview 출력을 XLinkOut의 입력에 연결합니다.
        # link()는 노드 간 단방향 데이터 흐름을 정의합니다.
        # 연결 후 VPU에서 실행되면: 카메라 프레임 -> USB XLink -> 호스트 큐
        # 이 연결이 없으면 카메라 데이터가 호스트에 전달되지 않습니다.
        cam_rgb.preview.link(xout_rgb.input)

        # =====================================================================
        # 스테레오 깊이 파이프라인 (선택적 활성화)
        # =====================================================================
        # 스테레오 깊이(Stereo Depth) 원리 상세 설명:
        #
        # 1. 기본 원리 - 삼각측량(Triangulation):
        #    좌/우 두 카메라가 동일한 3D 점을 서로 다른 시점에서 관측합니다.
        #    두 관측 사이의 수평 변위(시차/disparity)로부터 깊이를 역산합니다.
        #
        # 2. 시차(Disparity) 계산:
        #    왼쪽 이미지의 각 픽셀에 대해, 오른쪽 이미지에서 대응점을 찾습니다.
        #    두 대응점 사이의 수평 거리(픽셀 단위)가 시차입니다.
        #    OAK-D의 StereoDepth 노드는 Semi-Global Matching(SGM) 알고리즘을
        #    하드웨어로 구현하여 실시간 시차 계산을 수행합니다.
        #
        # 3. 깊이 변환 공식:
        #    depth(mm) = baseline(mm) * focal_length(px) / disparity(px)
        #    - baseline: 좌/우 카메라 간 물리적 거리 (OAK-D: 약 75mm)
        #    - focal_length: 카메라 초점 거리 (캘리브레이션 데이터에서 획득)
        #    - disparity: 시차 값 (클수록 가까움, 작을수록 멂)
        #
        # 4. 깊이맵 특성:
        #    - 데이터 타입: uint16 (부호 없는 16비트 정수)
        #    - 단위: 밀리미터 (mm)
        #    - 예: depth[y,x] = 1500 -> 해당 픽셀 위치의 물체가 1.5m 거리에 있음
        #    - 값이 0인 픽셀: 깊이를 계산할 수 없는 영역
        #      (텍스처 없는 벽면, 반사체, 가려진 영역 등)
        #    - 최대 측정 거리: 약 10~20m (환경 조건에 따라 다름)
        #    - 최소 측정 거리: 약 20~35cm (확장 디스패리티 활성화 시 더 가까워짐)
        #
        # OAK-D의 스테레오 깊이 처리는 VPU에서 하드웨어 가속으로 수행되므로
        # 호스트 CPU 부하 없이 실시간 깊이맵을 생성할 수 있습니다.
        # =====================================================================
        if self._enable_depth:
            # --- 좌측 모노 카메라 노드 ---
            # OAK-D의 좌측 흑백(Mono) 카메라를 설정합니다.
            # OV9282 센서: 1MP, 글로벌 셔터
            # 글로벌 셔터는 모든 픽셀을 동시에 노출하므로,
            # 롤링 셔터 대비 움직이는 물체 촬영 시 왜곡(젤리 현상)이 없습니다.
            # 스테레오 매칭에서 시간 동기화가 중요하므로 글로벌 셔터가 적합합니다.
            mono_left = pipeline.create(dai.node.MonoCamera)

            # --- 우측 모노 카메라 노드 ---
            mono_right = pipeline.create(dai.node.MonoCamera)

            # --- 스테레오 깊이 처리 노드 ---
            # StereoDepth 노드: 좌/우 모노 이미지를 받아 시차맵/깊이맵을 계산합니다.
            # 내부적으로 Semi-Global Matching(SGM) 알고리즘의 하드웨어 구현을 사용합니다.
            # SGM은 각 픽셀의 시차를 전역적 에너지 최소화로 결정하여,
            # 블록 매칭(BM) 대비 경계 영역에서 더 정확한 결과를 줍니다.
            stereo = pipeline.create(dai.node.StereoDepth)

            # 좌측 모노 카메라 해상도 설정: 400P (640x400)
            # 깊이 계산에는 고해상도가 반드시 필요하지 않으며,
            # 낮은 해상도는 처리 속도를 높이고 USB 대역폭을 절약합니다.
            # 사용 가능한 해상도:
            #   THE_400_P = 640x400  (기본, 성능 효율적)
            #   THE_480_P = 640x480
            #   THE_720_P = 1280x720 (고해상도 깊이맵 필요 시)
            #   THE_800_P = 1280x800
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            # setCamera("left"): 물리적 좌측 카메라 소켓(CAM_B)에 매핑합니다.
            # OAK-D 보드의 카메라 소켓 배치:
            #   CAM_A = 중앙 (RGB), CAM_B = 좌측 (Mono), CAM_C = 우측 (Mono)
            mono_left.setCamera("left")

            # 우측 모노 카메라도 동일한 해상도로 설정합니다.
            # 좌/우 해상도가 다르면 스테레오 매칭이 올바르게 동작하지 않습니다.
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            # setCamera("right"): 물리적 우측 카메라 소켓(CAM_C)에 매핑합니다.
            mono_right.setCamera("right")

            # 스테레오 깊이 프리셋(preset) 설정
            # HIGH_DENSITY 프리셋의 내부 설정:
            #   - 서브픽셀 디스패리티(Sub-pixel disparity) 활성화:
            #     정수 단위가 아닌 소수점 단위의 시차 계산으로 정밀도 향상
            #   - 확장 디스패리티(Extended disparity) 활성화:
            #     최소 측정 거리를 줄여 가까운 물체도 깊이 측정 가능
            #   - 중앙값 필터(Median filter) 적용:
            #     깊이맵의 노이즈를 줄여 부드러운 결과 생성
            #   - 결과: 대부분의 영역에서 깊이값을 얻을 수 있지만, 노이즈가 약간 있음
            #
            # HIGH_ACCURACY 프리셋과의 차이:
            #   - HIGH_ACCURACY는 신뢰도 임계값을 높게 설정하여
            #     확실한 영역에서만 깊이를 출력 (빈 영역 많음, 노이즈 적음)
            #   - 화재 감지 등 넓은 영역의 깊이 정보가 필요한 경우 HIGH_DENSITY가 적합
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

            # 깊이맵을 RGB 카메라 좌표계에 정렬 (Depth-to-RGB Alignment)
            #
            # 정렬 활성화 시 (OAKD_DEPTH_ALIGN_TO_RGB = True):
            #   - RGB 이미지의 각 픽셀(x,y)에 대응하는 깊이값 depth[y,x]를
            #     직접 얻을 수 있습니다.
            #   - 포즈 추정 결과의 관절 좌표(x,y)로 바로 깊이를 조회 가능
            #   - 예: 손목 좌표 (wx, wy) -> depth[wy, wx] = 1200mm (1.2m 거리)
            #
            # 정렬 비활성화 시:
            #   - 깊이맵이 좌측 모노 카메라 시점으로 출력됩니다.
            #   - RGB와 깊이 간 픽셀 대응을 위해 별도의 좌표 변환이 필요합니다.
            #   - 카메라 내부/외부 파라미터를 이용한 re-projection 필요
            #
            # CAM_A = RGB 카메라 소켓 (OAK-D에서 중앙 카메라)
            if OAKD_DEPTH_ALIGN_TO_RGB:
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

            # --- 노드 간 연결 (좌/우 모노 -> 스테레오 깊이) ---
            # 좌측 모노카메라의 출력(out)을 StereoDepth의 좌측 입력(left)에 연결합니다.
            # StereoDepth 노드는 좌/우 입력이 모두 연결되어야 동작합니다.
            mono_left.out.link(stereo.left)
            # 우측 모노카메라의 출력(out)을 StereoDepth의 우측 입력(right)에 연결합니다.
            mono_right.out.link(stereo.right)

            # --- XLinkOut 노드 생성 - 깊이맵을 호스트 PC로 전송 ---
            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")  # 스트림 이름: "depth"

            # StereoDepth의 깊이 출력을 XLinkOut에 연결합니다.
            # stereo.depth: uint16 형식의 깊이맵 출력 (단위: 밀리미터)
            #   - 각 픽셀 값 = 해당 위치의 물체까지 거리(mm)
            #   - 예: depth[y,x] = 1500 -> 1.5m 거리
            # stereo.disparity: 시차맵 출력 (시각화에 더 적합하지만, 거리 계산에는 depth 사용)
            # stereo.rectifiedLeft/Right: 정류된 좌/우 이미지 (디버깅/검증용)
            stereo.depth.link(xout_depth.input)

        # =====================================================================
        # 파이프라인을 OAK-D 디바이스에 업로드하고 실행 시작
        # =====================================================================
        # 파이프라인 객체를 인스턴스 변수로 저장합니다.
        # Python의 가비지 컬렉터가 pipeline을 수거하면 내부 노드 참조가 해제되므로,
        # 인스턴스 변수로 유지하여 디바이스 사용 중 참조가 끊기지 않도록 합니다.
        self._pipeline = pipeline

        # dai.Device(pipeline): OAK-D 디바이스에 USB로 연결하고, 파이프라인을
        # VPU 펌웨어로 컴파일하여 업로드합니다.
        # 이 호출이 성공하면 카메라 스트리밍이 즉시 시작됩니다.
        #
        # 내부 동작 순서:
        # 1. USB를 통해 연결 가능한 OAK-D 디바이스를 탐색
        # 2. 디바이스 부트로더에 연결
        # 3. 파이프라인을 바이너리로 직렬화
        # 4. VPU 펌웨어와 함께 디바이스로 업로드
        # 5. VPU에서 파이프라인 실행 시작
        # 6. XLink 통신 채널 수립
        #
        # 연결 가능한 OAK-D가 없으면 RuntimeError가 발생합니다.
        self._device = dai.Device(pipeline)

        # =====================================================================
        # 호스트 측 출력 큐 생성
        # =====================================================================
        # getOutputQueue(): XLink를 통해 디바이스에서 수신되는 프레임을 버퍼링하는
        # 큐를 호스트 측에 생성합니다.
        #
        # 파라미터 상세 설명:
        #
        # name="rgb":
        #   파이프라인에서 정의한 XLinkOut의 streamName과 정확히 일치해야 합니다.
        #   일치하지 않으면 해당 스트림의 데이터를 수신할 수 없습니다.
        #
        # maxSize=4:
        #   큐의 최대 용량 (프레임 수). 이 수만큼의 프레임을 호스트 메모리에 버퍼링합니다.
        #   값이 작을수록 (예: 1~2):
        #     - 지연(latency)이 줄어듦 (항상 최신 프레임에 가까움)
        #     - 호스트 처리가 잠시 느려지면 프레임 드롭 발생
        #   값이 클수록 (예: 8~16):
        #     - 프레임 드롭이 줄어듦
        #     - 오래된 프레임이 누적되어 지연이 증가할 수 있음
        #     - 메모리 사용량 증가
        #   4는 실시간 처리와 안정성 사이의 균형점입니다.
        #
        # blocking=False (비차단 모드):
        #   큐가 가득 찼을 때의 동작을 정의합니다.
        #   False (non-blocking, 본 프로젝트 사용):
        #     - 큐가 가득 차면 가장 오래된 프레임을 자동 폐기하고 새 프레임을 저장
        #     - 항상 최신에 가까운 프레임을 얻을 수 있어 실시간 처리에 적합
        #     - tryGet() 호출 시 큐가 비어있으면 None 반환 (대기하지 않음)
        #   True (blocking, 녹화 등에 사용):
        #     - 큐에 빈 공간이 생길 때까지 디바이스 측에서 대기
        #     - 모든 프레임을 빠짐없이 처리해야 하는 경우에 사용
        #     - 호스트 처리가 느리면 디바이스까지 백프레셔(backpressure) 전파
        self._rgb_queue = self._device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )

        # 깊이 스트림 큐도 동일한 방식으로 생성합니다 (깊이가 활성화된 경우에만).
        # RGB 큐와 독립적으로 동작하므로, 두 스트림의 프레임이 1:1 동기화되지는 않습니다.
        # 정확한 동기화가 필요한 경우 dai.node.Sync 노드를 사용할 수 있습니다.
        if self._enable_depth:
            self._depth_queue = self._device.getOutputQueue(
                name="depth", maxSize=4, blocking=False
            )

        # 초기화 성공 메시지를 콘솔에 출력합니다.
        # 해상도, FPS, 깊이 활성화 여부를 포함하여 현재 설정을 확인할 수 있습니다.
        print(f"[OK] OAK-D started: {OAKD_RESOLUTION_W}x{OAKD_RESOLUTION_H} @ {OAKD_FPS}fps"
              + (" + Depth" if self._enable_depth else ""))

    def _start_webcam(self):
        """
        일반 USB 웹캠을 OpenCV VideoCapture로 초기화하고 시작합니다.

        이 메서드는 다음 두 가지 상황에서 호출됩니다:
        1. 사용자가 명시적으로 --webcam 옵션을 지정한 경우
        2. OAK-D 초기화가 실패하여 자동 폴백된 경우

        웹캠 모드의 특성:
        - RGB 프레임만 제공됩니다 (깊이 데이터 없음).
        - cv2.VideoCapture를 사용하여 프레임을 캡처합니다.
        - 해상도/FPS 설정은 "요청(request)"이며, 보장되지 않습니다.
          실제 적용 여부는 카메라 하드웨어와 드라이버에 따라 다릅니다.
          웹캠이 요청한 값을 지원하지 않으면 가장 가까운 지원 값이 적용됩니다.
        - Windows에서는 MSMF(Media Foundation) 또는 DirectShow 백엔드가 사용됩니다.

        Raises
        ------
        RuntimeError
            지정한 webcam_id의 카메라를 열 수 없는 경우 발생합니다.
            원인: 카메라 미연결, 권한 문제, 다른 프로세스가 점유 중
        """
        # cv2.VideoCapture: OpenCV의 비디오 캡처 인터페이스
        # 정수를 전달하면 시스템 카메라 인덱스로 해석됩니다.
        #   0 = 첫 번째 카메라 (보통 내장 웹캠)
        #   1 = 두 번째 카메라 (보통 외부 USB 웹캠)
        # 문자열을 전달하면 비디오 파일 경로 또는 RTSP URL로 해석됩니다.
        self._cap = cv2.VideoCapture(self._webcam_id)

        # isOpened(): 카메라가 성공적으로 열렸는지 확인합니다.
        # False인 경우:
        #   - 해당 인덱스의 카메라가 시스템에 존재하지 않음
        #   - 카메라 접근 권한이 없음 (Windows: 카메라 개인 정보 설정)
        #   - 다른 애플리케이션이 이미 카메라를 점유 중
        #   - 드라이버 문제
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {self._webcam_id}")

        # 카메라 속성 설정 시도 (하드웨어가 지원하는 경우에만 실제로 적용됨)
        # set() 메서드는 성공 여부를 bool로 반환하지만, 여기서는 무시합니다.
        # 이유: 정확한 해상도가 아니더라도 가장 가까운 값으로 동작하면 충분하기 때문.
        #
        # CAP_PROP_FRAME_WIDTH: 프레임 가로 해상도 요청
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, OAKD_RESOLUTION_W)
        # CAP_PROP_FRAME_HEIGHT: 프레임 세로 해상도 요청
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OAKD_RESOLUTION_H)
        # CAP_PROP_FPS: 프레임 레이트 요청
        self._cap.set(cv2.CAP_PROP_FPS, OAKD_FPS)

        # 실제 적용된 값을 다시 읽어와 확인합니다.
        # 요청한 값과 다를 수 있으므로, 실제 값을 로그에 출력합니다.
        # 예: 640x480 @ 30fps를 요청했지만, 웹캠이 1280x720만 지원하면 그 값이 반환됨
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[OK] Webcam started: {w}x{h} @ {fps:.0f}fps")

    # =========================================================================
    # 프레임 획득 메서드
    # =========================================================================

    def get_frame(self):
        """
        현재 활성화된 카메라 소스에서 프레임을 가져옵니다.

        이 메서드는 카메라 모드(OAK-D / 웹캠)에 관계없이 동일한 인터페이스를
        제공하는 퍼사드(facade) 메서드입니다. 호출자는 내부 구현을 알 필요 없이
        항상 동일한 방식으로 프레임을 요청할 수 있습니다.

        OAK-D 모드에서는 tryGet() (비차단)을 사용하므로, 새 프레임이 아직
        도착하지 않았으면 (None, None)을 반환합니다.
        웹캠 모드에서는 read() (차단)을 사용하므로, 보통 항상 프레임을 반환합니다.

        메인 루프에서의 권장 사용 패턴:
            while True:
                bgr, depth = camera.get_frame()
                if bgr is None:
                    continue  # 프레임이 아직 없으면 다음 반복으로
                # bgr를 이용한 처리...

        Returns
        -------
        tuple (bgr_frame, depth_map)
            bgr_frame : numpy.ndarray 또는 None
                BGR 색상 순서의 이미지 프레임
                - shape: (H, W, 3), dtype: uint8
                - 값 범위: 0~255
                - OpenCV 함수들과 직접 호환 (imshow, imwrite 등)
                프레임을 가져올 수 없는 경우 None을 반환합니다.
                None 반환 사유:
                - OAK-D: tryGet()에서 아직 새 프레임이 도착하지 않은 경우
                - 웹캠: read() 실패 (카메라 연결 끊김, 스트림 종료 등)

            depth_map : numpy.ndarray 또는 None
                깊이맵 프레임
                - shape: (H, W), dtype: uint16
                - 단위: 밀리미터 (mm)
                - 예: depth_map[100, 200] = 1500 -> (200, 100) 위치의 물체가 1.5m 거리
                - 값 0 = 깊이 계산 불가 영역
                None 반환 사유:
                - 깊이 스트림 비활성화 (enable_depth=False)
                - 웹캠 모드 (스테레오 카메라 없음)
                - OAK-D에서 아직 깊이 프레임이 도착하지 않은 경우
        """
        if self._use_webcam:
            return self._get_webcam_frame()
        return self._get_oakd_frame()

    def _get_oakd_frame(self):
        """
        OAK-D에서 RGB 프레임과 (선택적) 깊이맵을 가져옵니다.

        DepthAI 큐에서 데이터를 가져오는 세 가지 방법:
        ================================================

        1. tryGet() [비차단/Non-blocking] -- 본 프로젝트에서 사용:
           - 큐에 데이터가 있으면 즉시 가장 오래된 데이터를 꺼내서 반환합니다.
           - 큐가 비어있으면 대기하지 않고 즉시 None을 반환합니다.
           - 메인 루프가 블로킹되지 않으므로 UI 응답성이 유지됩니다.
           - 실시간 처리에 적합: 프레임이 없으면 이전 결과를 재사용하거나 스킵
           - 반환 타입: dai.ImgFrame 또는 None

        2. get() [차단/Blocking]:
           - 큐에 데이터가 있으면 즉시 반환합니다.
           - 큐가 비어있으면 새 데이터가 도착할 때까지 현재 스레드를 블로킹합니다.
           - 타임아웃 없이 무한 대기할 수 있으므로 주의 필요
           - 모든 프레임을 빠짐없이 처리해야 하는 경우에 사용 (비디오 녹화 등)
           - UI 루프에서 사용하면 프레임 대기 중 화면이 멈출 수 있습니다.

        3. has() [확인만]:
           - 큐에 데이터가 있는지 bool로만 확인합니다.
           - 데이터를 꺼내지 않습니다 (peek 개념).
           - tryGet() 호출 전에 확인용으로 사용할 수 있으나,
             멀티스레드 환경에서는 경쟁 조건(race condition)에 주의해야 합니다.
             (has() 확인 후 tryGet() 사이에 다른 스레드가 데이터를 가져갈 수 있음)

        RGB-깊이 동기화에 대한 참고사항:
        ================================
        RGB 큐와 깊이 큐는 독립적인 XLink 스트림이므로, 프레임이 정확히
        동기화되지 않을 수 있습니다. 즉:
        - RGB 프레임은 도착했지만 대응하는 깊이 프레임은 아직 없을 수 있음
        - 깊이 프레임이 RGB보다 약간 지연될 수 있음 (스테레오 매칭 처리 시간)
        정밀한 동기화가 필요한 경우 dai.node.Sync 노드 또는 타임스탬프 기반
        매칭을 사용해야 합니다.

        Returns
        -------
        tuple (bgr_frame, depth_map)
            bgr_frame : numpy.ndarray (H x W x 3, uint8, BGR) 또는 None
            depth_map : numpy.ndarray (H x W, uint16, mm 단위) 또는 None
        """
        # RGB 큐에서 비차단 방식으로 프레임을 시도합니다.
        # tryGet()은 큐에서 가장 오래된(먼저 도착한) 프레임을 꺼냅니다.
        # 아직 새 프레임이 도착하지 않았으면 None이 반환됩니다.
        rgb_data = self._rgb_queue.tryGet()
        if rgb_data is None:
            # 프레임이 없는 경우: 호출자에게 (None, None) 반환
            # 호출자는 이 경우를 처리해야 합니다:
            #   - 이전 프레임 재사용 (화면 유지)
            #   - continue로 다음 루프 반복
            #   - 또는 짧은 sleep 후 재시도
            return None, None

        # getCvFrame(): DepthAI의 ImgFrame 객체를 OpenCV 호환 numpy 배열로 변환합니다.
        # 내부 동작:
        #   1. VPU에서 USB를 통해 수신된 바이너리 데이터를 numpy 배열로 변환
        #   2. planar(CHW) 형식이면 interleaved(HWC) 형식으로 자동 변환
        #   3. BGR 색상 순서의 numpy.ndarray 반환
        # 반환값 특성:
        #   - shape: (H, W, 3), H=OAKD_RESOLUTION_H, W=OAKD_RESOLUTION_W
        #   - dtype: uint8 (0~255)
        #   - 색상 순서: BGR (OpenCV 기본)
        #   - 메모리 복사가 발생하므로, 원본 ImgFrame과 독립적인 데이터입니다.
        bgr = rgb_data.getCvFrame()

        # 깊이맵 가져오기 (활성화된 경우)
        depth = None
        if self._enable_depth and self._depth_queue is not None:
            # 깊이 큐에서도 비차단 방식으로 시도합니다.
            # RGB와 깊이는 독립적인 스트림이므로, 동기화가 완벽하지 않을 수 있습니다.
            # 즉, RGB 프레임은 있지만 대응하는 깊이 프레임은 아직 없을 수 있습니다.
            # 이 경우 depth는 None으로 유지되며, bgr만 반환됩니다.
            depth_data = self._depth_queue.tryGet()
            if depth_data is not None:
                # getFrame(): 원시 프레임 데이터를 numpy 배열로 변환합니다.
                # getCvFrame()과의 차이:
                #   - getCvFrame(): 3채널 BGR 이미지용, HWC 형식 변환 포함
                #   - getFrame(): 단일 채널 또는 원시 데이터용, 형식 변환 없음
                # 깊이맵 특성:
                #   - shape: (H, W), 2D 배열 (단일 채널)
                #   - dtype: uint16 (부호 없는 16비트 정수)
                #   - 단위: 밀리미터 (mm)
                #   - 각 픽셀 값 = 해당 위치 물체까지의 거리
                #   - 예: depth[200, 300] = 2500 -> (300, 200) 위치 물체가 2.5m 거리
                #   - 값 0: 깊이 계산 불가 (텍스처 없는 영역, 반사, 가림 등)
                #   - 유효 범위: 약 200mm ~ 20000mm (0.2m ~ 20m)
                depth = depth_data.getFrame()

        return bgr, depth

    def _get_webcam_frame(self):
        """
        일반 USB 웹캠에서 프레임을 읽습니다.

        cv2.VideoCapture.read()는 차단(blocking) 호출이지만,
        웹캠 드라이버의 내부 버퍼에서 바로 프레임을 반환하므로
        실질적으로 1/FPS 이하의 시간 안에 완료됩니다.

        OAK-D의 tryGet()과의 동작 차이:
        - tryGet(): 프레임 없으면 즉시 None 반환 (비차단)
        - read(): 프레임이 준비될 때까지 대기 후 반환 (차단)
        따라서 웹캠 모드에서는 get_frame()이 거의 항상 프레임을 반환합니다.
        (None은 카메라 연결 끊김 같은 오류 상황에서만 반환됨)

        웹캠에서는 스테레오 카메라가 없으므로 깊이 데이터를 얻을 수 없습니다.
        따라서 depth는 항상 None으로 반환됩니다.

        Returns
        -------
        tuple (frame, None)
            frame : numpy.ndarray (H x W x 3, uint8, BGR) 또는 None
                - 성공: BGR 컬러 프레임
                - 실패: None (카메라 연결 끊김, 스트림 종료)
            두 번째 요소는 항상 None (웹캠은 깊이 데이터 미지원)
        """
        # read(): 다음 프레임을 캡처합니다.
        # 반환값:
        #   ret (bool): 프레임 캡처 성공 여부
        #     True = 프레임을 성공적으로 읽음
        #     False = 카메라 연결 끊김, 비디오 파일 끝, 기타 오류
        #   frame (numpy.ndarray): BGR 컬러 이미지
        #     shape: (H, W, 3), dtype: uint8
        #     ret=False인 경우 frame은 None 또는 빈 배열
        ret, frame = self._cap.read()
        if not ret:
            # 프레임 캡처 실패: 카메라 연결 끊김, USB 케이블 분리, 드라이버 오류 등
            # 호출자에게 (None, None)을 반환하여 오류 상황을 알립니다.
            return None, None
        # 웹캠은 깊이 데이터를 제공할 수 없으므로 두 번째 반환값은 항상 None
        return frame, None

    # =========================================================================
    # 리소스 해제 메서드
    # =========================================================================

    def stop(self):
        """
        카메라를 정지하고 모든 리소스를 해제합니다.

        이 메서드는 프로그램 종료 시 또는 카메라 재시작 전에 반드시 호출해야 합니다.

        리소스를 해제하지 않을 경우의 문제:
        - OAK-D: USB 연결이 유지되어 다음 실행 시 "디바이스 사용 중" 오류 발생
          물리적으로 USB를 뽑았다 꽂아야 해결될 수 있음
        - 웹캠: 카메라가 점유된 상태로 남아 다른 애플리케이션에서 접근 불가
          예: Zoom, Teams 등에서 "카메라를 사용할 수 없습니다" 오류

        이 메서드는 멱등성(idempotent)을 가집니다:
        - 여러 번 호출해도 안전합니다.
        - 이미 해제된 리소스에 대해 다시 호출해도 에러가 발생하지 않습니다.
        - None 체크 후 close()/release()를 호출하므로 이중 해제 문제가 없습니다.

        권장 사용 패턴 (try-finally):
            camera = OAKDCamera()
            camera.start()
            try:
                while True:
                    frame, depth = camera.get_frame()
                    ...
            finally:
                camera.stop()  # 예외 발생 시에도 반드시 리소스 해제
        """
        # OAK-D 디바이스 연결 종료
        if self._device is not None:
            # close(): USB 연결을 종료하고 VPU 파이프라인을 정지합니다.
            # 내부적으로 XLink 스트림을 닫고, VPU를 리셋 상태로 전환합니다.
            # 이후 디바이스를 다시 사용하려면 새 dai.Device 객체를 생성해야 합니다.
            self._device.close()
            self._device = None  # 참조 해제 (가비지 컬렉션 허용)

        # 웹캠 VideoCapture 해제
        if self._cap is not None:
            # release(): 카메라 디바이스를 운영체제에 반환합니다.
            # 이후 같은 카메라를 이 프로세스 또는 다른 프로세스에서 사용할 수 있게 됩니다.
            # release() 없이 프로세스가 종료되면 OS가 자동 회수하지만,
            # 명시적 해제가 안정적입니다.
            self._cap.release()
            self._cap = None     # 참조 해제

        # 시작 상태 플래그 초기화
        # stop() 후 다시 start()를 호출하여 카메라를 재시작할 수 있습니다.
        self._started = False
        print("[OK] Camera stopped.")

    # =========================================================================
    # 상태 확인 속성 (Properties)
    # =========================================================================

    @property
    def is_webcam(self):
        """
        현재 웹캠 모드로 동작 중인지 확인합니다.

        이 속성은 읽기 전용(read-only)이며, 내부 상태를 외부에 노출합니다.

        Returns
        -------
        bool
            True: 웹캠 모드 (일반 USB 카메라 사용 중)
            False: OAK-D 모드 (DepthAI 파이프라인 사용 중)

        참고:
        - start() 호출 전에는 __init__()에서 설정한 초기값을 반환합니다.
        - OAK-D 초기화 실패로 인해 자동 폴백된 경우, start() 이후
          원래 False였던 값이 True로 변경되어 반환됩니다.
        - 이 값으로 깊이 데이터 사용 가능 여부를 간접적으로 판단할 수 있습니다:
          if camera.is_webcam: depth는 항상 None
        """
        return self._use_webcam

    @property
    def is_started(self):
        """
        카메라가 현재 시작된 상태인지 확인합니다.

        Returns
        -------
        bool
            True: start()가 성공적으로 완료되어 get_frame() 호출 가능 상태
            False: 아직 start()가 호출되지 않았거나 stop()으로 정지된 상태

        이 속성은 get_frame() 호출 전에 카메라 상태를 확인하는 데 사용할 수 있습니다:
            if camera.is_started:
                frame, depth = camera.get_frame()
            else:
                print("카메라가 아직 시작되지 않았습니다.")
        """
        return self._started
