from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import sys
import numpy as np
import cv2 as cv
from model_dict import get_model_dict

model_dict = get_model_dict()

ARG_MODEL = sys.argv[1]
ARG_IMG_PATH = sys.argv[2]

model_info = model_dict[ARG_MODEL]

print(model_info)

model = load_model(model_info['path'])

img = cv.imread(ARG_IMG_PATH).astype(np.float32) / 255
h, w, c = img.shape
img = img.reshape(-1, h, w, c)

pred = model.predict(img)

img_out = np.rint(pred[0] * 255)

cv.imwrite('out.png', img_out)