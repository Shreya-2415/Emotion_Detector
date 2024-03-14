import tkinter as tk
from tkinter import filedialog
from tkinter import *
import pickle
from keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

#download haarcascade_frontalface_default from "https://github.com/opencv/opencv/tree/master/data/haarcascades"

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

    return model
    # try:
    #     model.load_weights(weights_file)
    #     print("Weights loaded successfully.")
    # except ValueError as e:
    #     print("Error loading weights:", str(e))


# def new_model(model):
#     return model

    # weights_new(weights_file, model)
    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # return model

# def weights_new(weights_file, model):
    # model.load_weights(weights_file)
    # try:
    #     model.load_weights(weights_file)
    #     print("Weights loaded successfully.")
    # except ValueError as e:
    #     print("Error loading weights:", str(e))

# def load_partial_weights(weights_file, model):
#     try:
#         model.load_weights(weights_file, by_name=True)  # Load weights of only the layers with matching names
#         print("Weights loaded successfully.")
#     except ValueError as e:
#         print("Error loading weights:", str(e))

# def load_partial_weights(weights_file, model):
#     try:
#         with open(weights_file, 'rb') as f:
#             model_weights = pickle.load(f)

#         # Loop through layers and assign weights if layer names match
#         for layer in model.layers:
#             if layer.name in model_weights:
#                 layer.set_weights(model_weights[layer.name])

#         print("Weights loaded successfully.")
#     except Exception as e:
#         print("Error loading weights:", str(e))

model = FacialExpressionModel("model_a.json", "model.weights.h5")
# load_partial_weights("model.weights.h5", model)
try:
    model.load_weights("model.weights.h5")
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background = '#CDCDCD')

label1 = Label(top, background= '#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json", "model.weights.h5")

EMOTIONS_LIST = ["Angry", "Happy","Disgust", "Fear", "Sad", "Neutral", "Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for(x,y,w,h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:, np.newaxis]))]
            print("Predicted Emotion is " + pred)
            label1.configure(foreground="#011638", text = pred)

    except:
        label1.configure(foreground="#011638", text = "Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top, text = "Detect Emotion", command= lambda: Detect(file_path), padx = 10, pady = 5)
    detect_b.configure(background="#364156", foreground= "white", font=('arial', 10, 'bold'))
    detect_b.place(relx= 0.79, rely= 0.46)

# def upload_image():
#     try:
#         file_path = filedialog.askopenfile()
#         uploaded = Image.open(file_path)
#         uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
#         im = ImageTk.PhotoImage(uploaded)

#         sign_image.configure(image= im)
#         sign_image.image = im
#         label1.configure(text = '')
#         show_Detect_button(file_path)
#     except:
#         pass
    
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            label1.configure(text='')
            show_Detect_button(file_path)
    except Exception as e:
        print(f"Error: {e}")

upload = Button(top, text = "Upload Image", command= upload_image, padx = 10, pady= 5)
upload.configure(background= "#364156", foreground= 'white', font=('arial', 20, 'bold'))
upload.pack(side = 'bottom', pady=50)
sign_image.pack(side = 'bottom', expand= 'True')
label1.pack(side='bottom', expand= 'True')
heading = Label(top, text = 'Emotion Detector', pady = 20, font=('arial', 25, 'bold'))
heading.configure(background= "#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()