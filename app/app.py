import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

model = load_model("models/asl_model.h5")
class_names = ['0','1','2','3','4','5','6','7','8','9']

st.title("ASL Real-Time Recognition")

class ASLTransformer(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        x1,y1,x2,y2 = 100,100,300,300
        roi = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

        resized = cv2.resize(thresh,(64,64))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1,64,64,1)

        pred = model.predict(reshaped)
        label = class_names[np.argmax(pred)]

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,label,(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return img


webrtc_streamer(
    key="asl",
    video_transformer_factory=ASLTransformer
)