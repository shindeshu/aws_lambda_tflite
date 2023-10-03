FROM public.ecr.aws/lambda/python:3.8

COPY artifacts/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl .

RUN pip3 install tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl --no-cache-dir
RUN pip install requests Pillow

COPY artifacts/cats_and_dogs.tflite .
COPY artifacts/lambda_function.py .

CMD ["lambda_function.lambda_handler"]