from flask import Flask, render_template, request, url_for
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import time
import os
import glob

app = Flask(__name__)

def generate_latent_points(latent_dim, class_label, n_classes=10):
    x_input = np.random.randn(latent_dim)
    z_input = x_input.reshape(1, latent_dim)
    labels = np.array([class_label])
    return [z_input, labels]

@app.route('/')
def index_view():
    return render_template('index.html', title="Fashion MNIST GAN")

@app.route('/', methods=['POST'])
def gan_prediction():
    text = request.form['text'].lower()
    classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
          'sneaker', 'bag', 'ankle boot']
    if text not in classes:
        raise ValueError('Didn\'t give a valid input! Go back and try again.')

    model = load_model('./static/conditional_generator.h5')
    latent_points, labels = generate_latent_points(100, classes.index(text))
    # predict requested label on model
    X = model.predict([latent_points, labels])
    # rescaling
    X = (X + 1) / 2.0
    for file in [f for f in os.listdir('./static/') if f.startswith("result")]:
        os.remove(f'./static/{file}')
    filename = f'./static/result{int(time.time())}.png'
    plt.imsave(fname=filename, arr=X[0, :, :, 0], cmap='gray_r')
    return render_template('result.html', filename=filename)
