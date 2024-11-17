from UDP_Client import UDPClient
from threading import Thread, Lock
from time import sleep
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

udp_client = UDPClient()
udp_client.sendHello()

model_path = 'E:/UnityProjects/FBT/Server/pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable and lock to store the detection result
global_detection_result = None
result_lock = Lock()

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global global_detection_result
    with result_lock:
        global_detection_result = result
    print('pose landmarker result: {}'.format(result))

def landmarks():
    with result_lock:
        return global_detection_result

def draw_landmarks_on_image(image, detection_result):
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

def process(landmarker):
    cap = cv2.VideoCapture(0)
    print("Video capture started")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        with result_lock:
            result_copy = global_detection_result

        if result_copy:
            annotated_image = draw_landmarks_on_image(frame, result_copy)
            cv2.imshow('Annotated Image', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video capture stopped")

def start_udp_client():
    print("UDP client started")
    while True:
        with result_lock:
            result_copy = global_detection_result

        if result_copy:
            udp_client.sendMessage(str(result_copy))  # Ensure landmarks are correctly formatted

        sleep(0.1)

def main():
    print("Main function started")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        process_thread = Thread(target=process, args=(landmarker,))
        process_thread.start()
        print("Image processing thread started")

        udp_thread = Thread(target=start_udp_client)
        udp_thread.start()
        print("UDP client thread started")

        process_thread.join()
        udp_thread.join()

if __name__ == "__main__":
    main()