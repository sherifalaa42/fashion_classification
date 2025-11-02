#!/usr/bin/env python
# coding: utf-8



from PIL import Image
import os
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(224, 224))

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'clothing-model.tflite')
interpet = tflite.Interpreter(model_path=MODEL_PATH)
interpet.allocate_tensors()



input_index = interpet.get_input_details()[0]['index']
output_index = interpet.get_output_details()[0]['index']

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def predict(url):
    X = preprocessor.from_url(url)

    interpet.set_tensor(input_index, X)
    interpet.invoke()
    preds = interpet.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

