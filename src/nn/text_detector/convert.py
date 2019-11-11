
import tensorflow as tf
from model import Model

model = Model(480, 640)
model.load_weights('checkpoints/final.tf')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_tflite = converter.convert()
with open("model.tflite", "wb") as f:
	f.write(model_tflite)
