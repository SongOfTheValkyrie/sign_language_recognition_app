import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
hands = mp_hand.Hands()
mp_drawing_utils = mp.solutions.drawing_utils

# Load your respective model here
model = keras.models.load_model("saved_model/my_model")

targets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Get the hand landmarks object from the frame
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        # Uncomment to draw hand landmarks on the frame (can be totally removed if predicting w/o landmarks)
        #for hand_landmark in result.multi_hand_landmarks:
        #    mp_drawing_utils.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

        # Extract hand landmarks from the frame for prediction (can be removed if predicting w/o landmarks)
        hand_landmarks = []
        for hand_landmark in result.multi_hand_landmarks[0].landmark:
            hand_landmarks.extend([hand_landmark.x, hand_landmark.y, hand_landmark.z])

        # Here you can use any of the models to make predictions
        prediction = targets[np.argmax(model.predict(np.array([hand_landmarks])))]

        # Add prediction to frame, see the documentation to format the text
        cv2.putText(frame, "Sign: %s" % prediction,(50,50), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 225, ), 3)

    # Display the resulting frame
    cv2.imshow("SignNet", frame)
    
    # Press q to exit the application
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
