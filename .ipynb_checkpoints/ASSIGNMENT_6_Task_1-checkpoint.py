# Importing modules
import sys
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
import io
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from plot_keras_history import show_history
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model

# Model training 
# User can comment this model training (till model.save) and use the already saved model ("MNIST_model.h5") which is submitted (along 
# with the python scripts) by uploading it in the current working directory

# Loading the dataset and spliting
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
num_classes = 10
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, 'train input samples')
print(x_test.shape, 'test input samples')

y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
print(y_train.shape, 'train output samples')
print(y_test.shape, 'test output samples')

# Model
model = keras.Sequential()
model.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

show_history(history)

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy: {:5.2f}%".format(100*acc))
loss, acc = model.evaluate(x_train, y_train, verbose=2)
print("Train accuracy: {:5.2f}%".format(100*acc))

# Saving the model
model.save("MNIST_model.h5")
 

# Creating the fast api module
app = FastAPI()

# Getting the model path using the arguments (Format python3 <filename> <path>) 
# Run this command : python3 ASSIGNMENT_6_Task_1.py "MNIST_model.h5"
model_path = sys.argv[1]

# Function to load model
def load_model_(path: str):
    return load_model(path)
    

# Function to predict the digit
def predict_digit(model, data_point: list) -> str:
    # Preprocess the data
    data_point = np.array(data_point).reshape(1, 784)
    # Perform prediction
    prediction = model.predict(data_point)
    # Get the predicted digit
    digit = np.argmax(prediction)
    return str(digit)

# Added to display a message in the local host URL page
@app.get("/")
def read_root():
    return {"Building a FastAPI for MNIST digit prediction"}

# Creating the api endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #Load the model
    model = load_model_(model_path)
    
    # Read the uploaded image from the file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # Resize image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert image to numpy array
    img = np.array(image)
    # Normalize the image
    img = img.astype('float32') / 255.0
    # Flattening the image
    img_flat = img.flatten()
    # Make prediction
    digit = predict_digit(model, img_flat)
    return {"digit": digit}


# Running the FastAPI application using Uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)


# COMMAND TO ACCESS THE SWAGGER UI IN THE WEB SERVER
# http://localhost:8000/docs
