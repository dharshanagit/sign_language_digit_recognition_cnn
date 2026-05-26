import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/asl_model.h5")

# Class labels
class_names = ['0','1','2','3','4','5','6','7','8','9']

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # ROI box
    x1, y1 = 100, 100
    x2, y2 = 300, 300

    # Draw rectangle
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold
    _, thresh = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Resize
    img = cv2.resize(thresh, (64,64))

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1,64,64,1)

    # Prediction
    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    label = class_names[predicted_class]

    # Show prediction only if confidence high
    if confidence > 0.70:

        cv2.putText(
            frame,
            f"Prediction: {label}",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    # Show threshold image
    cv2.imshow("Threshold", thresh)

    # Show webcam
    cv2.imshow("ASL Digit Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()