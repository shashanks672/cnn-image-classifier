import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.h5")

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))

# VERY IMPORTANT FIX
img = 255 - img

img = img / 255.0
img = img.reshape(1,28,28,1)

prediction = model.predict(img)
print("Predicted:", np.argmax(prediction))