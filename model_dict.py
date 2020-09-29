def get_model_dict():
	return {
	'2x': {
		'path': 'models/upsample2x_model_13.h5',
	},
	'2x-small': {
		'path': 'models/upsample2x_model_1.h5',
	},
	'2x-large': {
		'path': 'models/upsample2x_model_14.h5',
	},
	'4x': {
		'path': 'models/upsample4x_model_6.h5',
	},
	'8x': {
		'path': 'models/upsample8x_model_3.h5',
	},
	'jpeg-verylow': {
		'path': 'models/jpeg-denoise-90-100_model_5.h5',
	},
	'jpeg-low':{
		'path': 'models/jpeg-denoise-75-89_model_3.h5',
	},
	'jpeg-medium':{
		'path': 'models/jpeg-denoise-55-74_model_3.h5',
	},
	'jpeg-high':{
		'path': 'models/jpeg-denoise-30-54_model_1.h5',
	},
	'jpeg-veryhigh':{
		'path': 'models/jpeg-denoise-1-29_model_1.h5',
	},
	'jpeg-quality-classifier':{
		'path': 'models/jpeg-quality-classifier_model_1.h5',
	}
}