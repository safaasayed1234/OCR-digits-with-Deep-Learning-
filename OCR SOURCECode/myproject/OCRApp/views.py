from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pytesseract
import joblib

model = joblib.load('OCRApp/digit_classifier_model.joblib')


def index(request):
    return render(request, 'OCRApp/index.html')


def predict(request):
    prediction = None
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded file
        image_file = request.FILES['image']
        print("Image uploaded:", image_file.name)

        # Open the image
        image = Image.open(image_file)
        print("Image opened:", image)

        # Preprocess the image (convert to grayscale, resize, normalize)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to the input size the model expects
        image_array = np.array(image) / 255.0  # Normalize the pixel values
        print("Processed image array:", image_array)

        image_array = image_array.reshape(1, 28, 28, 1)
        print("Reshaped image array:", image_array)

        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        print("Predicted digit:", predicted_digit)

        # Return the result to the template
        return render(request, 'OCRApp/index.html', {'prediction': predicted_digit})

    return render(request, 'OCRApp/index.html', {'prediction': prediction})

