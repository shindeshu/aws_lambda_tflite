from utils import Timer, load_image, postprocess
timer = Timer()
import tflite_runtime.interpreter as tflite
timer.end("Importing Libraries")
import argparse
url = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"

def load_tflite():
    interpreter = tflite.Interpreter(model_path='cats_and_dogs.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    return interpreter, (input_index, output_index)

def predict_with_tflite(interpreter, indexes, img):
    interpreter.set_tensor(indexes[0], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(indexes[1])
    return postprocess(preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action='store_true', help='Benchmark the speed.')
    args = parser.parse_args()

    model, indexes = load_tflite()
    timer.end("Loading model")

    if args.benchmark:
        img = load_image(url)
        timer.start()
        for i in range(100):
            preds = predict_with_tflite(model, indexes, img)
        timer.end(f"100 Inputs")
    else:
        img = load_image(url)
        preds = predict_with_tflite(model, indexes, img)
        print(preds)
    timer.end("End of script", script_end=True)