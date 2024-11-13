# classifier/views.py
from django.shortcuts import render, redirect
from .models import ImageUpload
from .forms import ImageUploadForm
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.save()  # Save the uploaded image

            # Process the image for prediction
            img_path = img_obj.image.path
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make predictions
            preds = model.predict(x)
            results = decode_predictions(preds, top=3)[0]

            return render(request, 'classifier/results.html', {
                'results': results,
                'image_url': img_obj.image.url  # Pass the image URL to the template
            })
    else:
        form = ImageUploadForm()
    return render(request, 'classifier/upload.html', {'form': form})
