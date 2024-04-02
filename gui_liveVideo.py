import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import Button, Label
import pickle
from keras.models import model_from_json
# from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    return model

model = FacialExpressionModel("model_a.json", "model.weights.h5")

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ["Happy", "Angry", "Disgust", "Fear", "Sad", "Neutral", "Surprise"]

video_capture = cv2.VideoCapture(0)

def Detect(video_capture):
    global Label_packed
    while True:
        ret, video_data = video_capture.read()
        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(col, 1.3, 5)
        try:
            for (x, y, w, h) in faces:
                cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
                fc = col[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
                # print("Predicted Emotion is " + pred)
                cv2.putText(video_data, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                label1.configure(foreground="#011638", text=pred)
        except Exception as e:
            print("Error:", e)
            label1.configure(foreground="#011638", text="Unable to detect")
        
        cv2.imshow("Video live", video_data)
        # if cv2.waitKey(10) == ord("a"):
        #     break
        if cv2.waitKey(100) & 0xFF == 27:  # Exit when ESC is pressed
            break

def show_Detect_button():
    detect_b = Button(top, text="Open the video", command=lambda: Detect(video_capture), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

show_Detect_button()

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()

# Release video capture when application closes
video_capture.release()


#-------------------------------------------------------------------

# import tkinter as tk
# from tkinter import filedialog, Label, Button
# import pickle
# from keras.models import model_from_json
# from PIL import Image, ImageTk
# import numpy as np
# import cv2
# import time

# def FacialExpressionModel(json_file, weights_file):
#     with open(json_file, "r") as json_file:
#         loaded_model_json = json_file.read()
#         model = model_from_json(loaded_model_json)
#     return model

# def on_closing():
#     global video_capture
#     if video_capture.isOpened():
#         video_capture.release()
#     top.destroy()

# model = FacialExpressionModel("model_a.json", "model.weights.h5")

# top = tk.Tk()
# top.geometry('800x600')
# top.title('Emotion Detector')
# top.configure(background='#CDCDCD')

# label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
# label1.pack()

# facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# EMOTIONS_LIST = ["Angry", "Happy", "Disgust", "Fear", "Sad", "Neutral", "Surprise"]

# video_capture = cv2.VideoCapture(0)

# def Detect(video_capture, label):
#     global Label_packed
#     while True:
#         ret, video_data = video_capture.read()
#         col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
#         faces = facec.detectMultiScale(col, 1.3, 5)
#         try:
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 fc = col[y:y+h, x:x+w]
#                 roi = cv2.resize(fc, (48, 48))
#                 pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
#                 print("Predicted Emotion is " + pred)
#                 label.configure(foreground="#011638", text=pred)
#                 # Introduce a delay of 1 second between each detection
#                 time.sleep(1)
#         except Exception as e:
#             print("Error:", e)
#             label.configure(foreground="#011638", text="Unable to detect")
        
#         img = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)

#         key = cv2.waitKey(10)
#         if key == 27:  # Escape key code
#             break

# def show_Detect_button():
#     show_Detect_button.configure(state='disabled')  # Disable the button while video is running
#     Detect(video_capture, label1)

# show_Detect_button = Button(top, text="Open the video", command=show_Detect_button, padx=10, pady=5)
# show_Detect_button.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
# show_Detect_button.pack()

# heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
# heading.configure(background="#CDCDCD", foreground="#364156")
# heading.pack()

# # Bind the on_closing function to the window close event
# top.protocol("WM_DELETE_WINDOW", on_closing)

# top.mainloop()

# # Release video capture when application closes
# video_capture.release()

import tkinter as tk
from tkinter import Button, Label
import numpy as np
import cv2
import time

def FacialExpressionModel(json_file, weights_file):
    # Load the facial expression recognition model
    pass  # Placeholder for loading model

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
