from server import Server
from threading import Thread, Lock
from time import sleep
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import time
import asyncio

server = Server('127.0.0.1', '9085')

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
            server.send_string(str(result_copy))  # Ensure landmarks are correctly formatted

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video capture stopped")




async def main():
    while True:
        request = server.receive()
        text_string = request.decode()
        # text_string = re.sub(r'(\r\n.?)+', r'\r\n', text_string)
        # text_string=text_string.strip('\r\n ')
        # text_string = re.sub('\s+',' ', text_string)
        # text = pygmalion.prompt(f"{personality} <START>/n What's your name?", MAX_LENGTH)
        # text = text.split("\n",1)[1]
        #emotions = emotion_analyzer.analyze(text_string)
        #translation = translator.translate(text_string, "jpn_Jpan", MAX_LENGTH)
        #r = {"text": translation, "emotions": emotions}
        #r = json.dumps(r, ensure_ascii=False)
        server.send_string("hello")


asyncio.run(main())