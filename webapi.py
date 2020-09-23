import flask
import cv2 as cv
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from model_dict import get_model_dict

model_dict = get_model_dict()

jpeg_denoise_trans = [None, 'jpeg-verylow', 'jpeg-low', 'jpeg-medium', 'jpeg-high', 'jpeg-veryhigh']
scale_trans = [None, None, '2x', None, '4x']

models = {}

for func, info in model_dict.items():
	models[func] = load_model(info['path'])

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return flask.render_template('index.html')

@app.route('/maxres', methods=['POST'])
def maxres():
	scale = int(flask.request.form.get('scale'))
	jpeg_denoise = int(flask.request.form.get('jpeg'))
	img = None

	if scale not in [1, 2, 4]:
		return 'Wrong parameters.', 400
	if jpeg_denoise not in [-1, 0, 1, 2, 3, 4, 5]:
		return 'Wrong parameters.', 400

	if flask.request.files.get('image'):
		image = flask.request.files['image'].read()

		nparr = np.fromstring(image, np.uint8)
		img = cv.imdecode(nparr, cv.IMREAD_COLOR)

	if type(img).__module__ == np.__name__:
		img = img.astype(np.float32) / 255
		h, w, c = img.shape
		img = img.reshape(-1, h, w, c)

		if jpeg_denoise in [1, 2, 3, 4, 5]:
			model = models[jpeg_denoise_trans[jpeg_denoise]]
			img = model.predict_on_batch(img)

		if scale in [2, 4]:
			model = models[scale_trans[scale]]
			img = model.predict_on_batch(img)

		out = np.rint(img[0] * 255)
		is_success, buffer = cv.imencode(".png", out)
		file_obj = io.BytesIO(buffer)
		file_obj.seek(0)

		return flask.send_file(file_obj, mimetype='image/PNG')

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()