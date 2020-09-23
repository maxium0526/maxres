import random
from tensorflow.keras.losses import MSE
from tensorflow.image import ssim, ssim_multiscale
import cv2 as cv

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def crop_image(img, size, num=1):
	h, w, b = img.shape
	out = []
	for i in range(num):
		r = random.randint(0, w-size)
		c = random.randint(0, h-size)
		out.append(img[c:c+size, r:r+size, :])
	return out

def jpeg_compress(img, quality):
	_ , encimg = cv.imencode('.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
	decimg = cv.imdecode(encimg, 1)
	return decimg

def SSIM(y_true, y_pred):
	return ssim(y_true, y_pred, 1)

def MS_SSIM(y_true, y_pred):
	return ssim_multiscale(y_true, y_pred, 1)