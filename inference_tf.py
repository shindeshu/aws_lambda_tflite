from utils import load_image, Timer, postprocess
timer = Timer()
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
timer.end("importing tensorflow")
import argparse
import numpy as np
from PIL import Image
import requests

url = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action='store_true', help='Benchmark the speed.')
    args = parser.parse_args()

    model = load_model("cats_and_dogs.h5")
    timer.end("Loading model")

    if args.benchmark:
        img = load_image(url)
        timer.start()
        for i in range(100):
            preds = postprocess(model.predict(img))
        timer.end(f"100 Inputs")
    else:
        img = load_image(url)
        preds = postprocess(model.predict(img).tolist())
        print(preds)
    timer.end("End of script", script_end=True)