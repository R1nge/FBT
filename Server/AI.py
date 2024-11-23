import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time

class AI:
    model_path = 'E:/UnityProjects/FBT/Server/pose_landmarker_full.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    def print_result(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        annotated_image = self.draw_landmarks_on_image(output_image.numpy_view(), result)
        #print('pose landmarker result: {}'.format(result))
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Annotated Image', annotated_image_bgr)
        cv2.waitKey(100)

    def draw_landmarks_on_image(self, image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def init(self):
        options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )

        self.PoseLandmarker = self.PoseLandmarker.create_from_options(options)

    def process(self):
        #print("Video capture started")
        cap = cv2.VideoCapture(0)
        while cap.isOpened:
            #print("Video capture started cap open")
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.PoseLandmarker.detect_async(image=mp_image,
                                             timestamp_ms=int((cv2.getTickCount() / cv2.getTickFrequency()) * 1000))

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Annotated Image', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video capture stopped")
