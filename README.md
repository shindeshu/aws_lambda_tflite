# aws_lambda_tflite
a toy example of deploying a tensorflow model using AWS Lambda


The model is trained on Kaggle on the Cats-vs-Dogs classification dataset. The notebook can be viewed here: [A Friendly Introduction to CNNs in Keras](https://www.kaggle.com/code/shindeshubham85/a-friendly-introduction-to-cnns-in-keras).

We download the model from kaggle as an .h5 file, convert it to tflite in the notebook `tflite_conversion.ipynb`. We benchmark the time taken by pure tensorflow vs. tflite in the inference scripts.

<detailed writeup soon.>