import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ===============================
# LOAD MODEL
# ===============================
model = load_model("models/asl_model.h5")

# ===============================
# CLASS LABELS
# MUST MATCH TRAINING ORDER
# ===============================
class_names = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

sentence = ""

# Prediction smoothing
pred_buffer = deque(maxlen=20)

stable_char = ""
stable_count = 0

# 🔥 LOWERED THRESHOLD
CONF_THRESHOLD = 0.75

STABLE_FRAMES = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ===============================
    # BIGGER ROI (IMPORTANT FOR DIGITS)
    # ===============================
    x1, y1 = 70, 70
    x2, y2 = 400, 400

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

    roi = frame[y1:y2, x1:x2]

    # ===============================
    # PREPROCESSING
    # ===============================
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Mild blur only
    gray = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    # Resize
    img = cv2.resize(
        gray,
        (64,64)
    )

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape
    img = img.reshape(1,64,64,1)

    # ===============================
    # PREDICTION
    # ===============================
    prediction = model.predict(
        img,
        verbose=0
    )[0]

    # Top prediction
    class_index = np.argmax(prediction)

    confidence = prediction[class_index]

    label = class_names[class_index]

    # ===============================
    # TOP 3 DEBUGGING
    # ===============================
    top3_idx = prediction.argsort()[-3:][::-1]

    top3_text = ""

    for idx in top3_idx:

        top3_text += f"{class_names[idx]}:{prediction[idx]:.2f} "

    # ===============================
    # CONFIDENCE CHECK
    # ===============================
    if confidence > CONF_THRESHOLD:

        pred_buffer.append(label)

        most_common = Counter(
            pred_buffer
        ).most_common(1)[0][0]

        # Stability logic
        if most_common == stable_char:

            stable_count += 1

        else:

            stable_char = most_common

            stable_count = 0

        # Add to sentence
        if stable_count >= STABLE_FRAMES:

            # Prevent duplicate spam
            if len(sentence) == 0 or sentence[-1] != stable_char:

                sentence += stable_char

            stable_count = 0

        # ===============================
        # DISPLAY
        # ===============================
        cv2.putText(
            frame,
            f"Prediction: {most_common}",
            (40,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (40,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Low Confidence",
            (40,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        pred_buffer.clear()

        stable_count = 0

        stable_char = ""

    # ===============================
    # TOP 3 DISPLAY
    # ===============================
    cv2.putText(
        frame,
        top3_text,
        (40,130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    # Sentence
    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (40,470),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    # Instructions
    cv2.putText(
        frame,
        "C = Clear | Q = Quit",
        (40,520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # Show
    cv2.imshow(
        "ASL Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('c'):
        sentence = ""

cap.release()

cv2.destroyAllWindows()