from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import pyautogui
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def mouse_move(detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  
  if len(hand_landmarks_list) == 1:
    points = hand_landmarks_list[0]
    wrist = points[4]
    screen_width, screen_height = pyautogui.size()
    move_x = 0.9272 * wrist.x + 0.3746 * wrist.y
    move_y = 0.9272 * wrist.y - 0.3746 * wrist.x
    pyautogui.moveTo(screen_width * move_x, screen_height * move_y)

# def mouse_click_left(detection_result):

# def mouse_click_right(detection_result):

cap = cv2.VideoCapture(1)
cap.set(3, 320)
cap.set(4, 240)

while True:
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)

  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
  detector = vision.HandLandmarker.create_from_options(options)

  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

  detection_result = detector.detect(image)

  mouse_move(detection_result)

  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  cv2.imshow('Hand Landmarker', annotated_image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
