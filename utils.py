import timeit
import requests
from PIL import Image
import numpy as np
from io import BytesIO

class Timer():
    def __init__(self, ):
        self.start_time = timeit.default_timer()
        self.init_time = timeit.default_timer()

    def start(self, ):
        self.start_time = timeit.default_timer()

    def end(self, message="", div=1.0, script_end=False):
        self.end_time = timeit.default_timer()
        if script_end:
            time_elapsed = (self.end_time - self.init_time)/div
        else:
            time_elapsed = (self.end_time - self.start_time)/div
        print(f"Time elapsed for {message}: {time_elapsed:.2f}")
        start_time = timeit.default_timer()

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