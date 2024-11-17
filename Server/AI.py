class AI:
  import cv2
  import mediapipe as mp
  import numpy as np
  from mediapipe.framework.formats import landmark_pb2
  from mediapipe import solutions

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
      pose_landmarks_list = detection_result.pose_landmarks
      annotated_image = np.copy(image)
    
      # Loop through the detected poses to visualize.
      for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
    
        # Draw the pose landmarks.
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