import tflite_runtime.interpreter as tflite
import timeit
import requests
from PIL import Image
import numpy as np
from io import BytesIO

def load_tflite():
    interpreter = tflite.Interpreter(model_path='cats_and_dogs.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    return interpreter, (input_index, output_index)

interpreter, indexes = load_tflite()

def predict_with_tflite(img):
    interpreter.set_tensor(indexes[0], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(indexes[1])[0].tolist()
    return postprocess(preds)

def load_image(path, from_url=True, process=True):
    """
    Custom preprocessing function. 
    """
    if from_url:
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    if process:
        img = img.resize((224,224), Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32)
        img = img * (1./255)
    return np.asarray([img])

def postprocess(preds: list):
    classes = ['cat', 'dog']
    return dict(zip(classes, preds))


def lambda_handler(event, context):
    url = event['url']
    X = load_image(url)
    result = predict_with_tflite(X)
    return result
