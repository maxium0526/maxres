import flask
import cv2 as cv
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from model_dict import get_model_dict

model_dict = get_model_dict()

jpeg_denoises = ['jpeg-none', 'jpeg-auto', 'jpeg-verylow', 'jpeg-low', 'jpeg-medium', 'jpeg-high', 'jpeg-veryhigh']
scales = ['1x', '2x', '4x']

models = {}

for func, info in model_dict.items():
	models[func] = load_model(info['path'])

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return flask.render_template('index.html')

@app.route('/maxres', methods=['POST'])
def maxres():
	scale = flask.request.form.get('scale')
	jpeg_denoise = flask.request.form.get('jpeg')
	img = None

	if scale not in scales:
		return 'Wrong parameters.', 400
	if jpeg_denoise not in jpeg_denoises:
		return 'Wrong parameters.', 400

	if flask.request.files.get('image'):
		image = flask.request.files['image'].read()

		nparr = np.fromstring(image, np.uint8)
		img = cv.imdecode(nparr, cv.IMREAD_COLOR)

	if type(img).__module__ == np.__name__:
		img = img.astype(np.float32) / 255
		h, w, c = img.shape
		img = img.reshape(-1, h, w, c)

		if jpeg_denoise in jpeg_denoises:
			if jpeg_denoise in ['jpeg-verylow', 'jpeg-low', 'jpeg-medium', 'jpeg-high', 'jpeg-veryhigh']:	
				model = models[jpeg_denoise]
				img = model.predict_on_batch(img)
			elif jpeg_denoise in ['jpeg-auto']:
				pass
			elif jpeg_denoise in ['jpeg-none']:
				pass

		if scale in scales:
			if scale in ['2x', '4x']:
				model = models[scale]
				img = model.predict_on_batch(img)
			elif scale in ['1x']:
				pass

		out = np.rint(img[0] * 255)
		is_success, buffer = cv.imencode(".png", out)
		file_obj = io.BytesIO(buffer)
		file_obj.seek(0)

		return flask.send_file(file_obj, mimetype='image/PNG')

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()