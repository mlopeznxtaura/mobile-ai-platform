"""
MediaPipe real-time vision tasks for mobile.
Face detection, pose estimation, hand tracking, object detection.
SDKs: MediaPipe, OpenCV
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not available. Install: pip install mediapipe")


@dataclass
class VisionResult:
    task: str
    detections: List[Dict[str, Any]]
    inference_ms: float
    frame_shape: tuple


class MediaPipeVision:
    """
    Real-time vision tasks via MediaPipe.
    Face detection, pose, hands, object detection — all on-device.
    """

    def __init__(self):
        if not MP_AVAILABLE:
            raise ImportError("mediapipe required. Install: pip install mediapipe")
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def detect_faces(self, frame: np.ndarray, min_confidence: float = 0.5) -> VisionResult:
        """Detect faces in a BGR frame."""
        import time
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face.FaceDetection(min_detection_confidence=min_confidence) as detector:
            results = detector.process(rgb)

        detections = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                detections.append({
                    "confidence": det.score[0],
                    "bbox": {
                        "x": bbox.xmin * w, "y": bbox.ymin * h,
                        "w": bbox.width * w, "h": bbox.height * h,
                    },
                })

        elapsed = (time.perf_counter() - t0) * 1000
        return VisionResult("face_detection", detections, elapsed, frame.shape)

    def estimate_pose(self, frame: np.ndarray) -> VisionResult:
        """Full body pose estimation (33 landmarks)."""
        import time
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(rgb)

        landmarks = []
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    "id": i, "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
                })

        elapsed = (time.perf_counter() - t0) * 1000
        return VisionResult("pose", landmarks, elapsed, frame.shape)

    def detect_hands(self, frame: np.ndarray) -> VisionResult:
        """Hand landmark detection (21 landmarks per hand)."""
        import time
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            results = hands.process(rgb)

        detected_hands = []
        if results.multi_hand_landmarks:
            for hand_lms, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand = {
                    "label": handedness.classification[0].label,
                    "confidence": handedness.classification[0].score,
                    "landmarks": [
                        {"id": i, "x": lm.x, "y": lm.y, "z": lm.z}
                        for i, lm in enumerate(hand_lms.landmark)
                    ],
                }
                detected_hands.append(hand)

        elapsed = (time.perf_counter() - t0) * 1000
        return VisionResult("hands", detected_hands, elapsed, frame.shape)

    def run_camera(self, task: str = "face_detection", camera_id: int = 0):
        """Run vision task on live camera feed."""
        cap = cv2.VideoCapture(camera_id)
        task_fn = {
            "face_detection": self.detect_faces,
            "pose": self.estimate_pose,
            "hands": self.detect_hands,
        }.get(task, self.detect_faces)

        print(f"[MediaPipe] Running {task} on camera {camera_id}. Press Q to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = task_fn(frame)
            cv2.putText(frame, f"{task}: {len(result.detections)} | {result.inference_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Mobile AI - {task}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
