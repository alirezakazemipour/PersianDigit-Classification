# Imports
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt

model = VGG16(include_top=False,
              weights="imagenet")
model = Model(inputs=model.inputs, outputs=model.layers[3].output)
model.summary()

img = load_img("samand.jpg")
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

result = model.predict(img)
output_size = result.shape[-1]
plt.figure()
for i in range(1, 1 + output_size):
    plt.subplot(8, 8, i)
    plt.imshow(result[0, :, :, i - 1], cmap="gray")
    plt.axis("off")
plt.savefig("figure.png")
plt.close("all")
