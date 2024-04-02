import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import Button, Label
import pickle
from keras.models import model_from_json
import numpy as np
import cv2
import time

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    return model

model = FacialExpressionModel("model_a.json", "model.weights.h5")

def detect_emotion():
    # Function to detect emotion from live video streaming
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    EMOTIONS_LIST = ["Happy", "Angry", "Disgust", "Fear", "Sad", "Neutral", "Surprise"]
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to open video capture.")
        return

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Process the region of interest (ROI) containing the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Placeholder for emotion prediction using the model
            predicted_emotion = EMOTIONS_LIST[np.random.randint(0, len(EMOTIONS_LIST))]  # Random prediction for demonstration

            # Display the predicted emotion
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Introduce a delay between each frame processing to control processing speed
        time.sleep(0.1)

    video_capture.release()
    cv2.destroyAllWindows()

# GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detection')
top.configure(background='#CDCDCD')

heading = Label(top, text='Emotion Detection', pady=20, font=('arial', 25, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

detect_button = Button(top, text="Start Detection", command=detect_emotion, padx=10, pady=5)
detect_button.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
detect_button.place(relx=0.4, rely=0.5)

top.mainloop()
