import cv2
import mediapipe as mp
import numpy as np

model_path = 'E:/UnityProjects/FBT/Server/pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the detection result
global_detection_result = None

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global global_detection_result
    global_detection_result = result
    print('pose landmarker result: {}'.format(result))

def draw_landmarks_on_image(image, detection_result):
    if detection_result.pose_landmarks:
        print("pose_landmarks structure:", type(detection_result.pose_landmarks))
        for idx, landmarks in enumerate(detection_result.pose_landmarks):
            print(f"Landmark set {idx} structure:", type(landmarks))
            for point in landmarks:
                print(f"Point structure:", type(point))
                cv2.circle(image, (int(point.x * image.shape[1]), int(point.y * image.shape[0])), 2, (0, 255, 0), -1)
    return image

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with PoseLandmarker.create_from_options(options) as landmarker:

    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process the image asynchronously
        landmarker.detect_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        if global_detection_result:
            # Draw landmarks on the image using the latest detection result
            annotated_image = draw_landmarks_on_image(frame, global_detection_result)

            # Display the annotated image
            cv2.imshow('Annotated Image', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()