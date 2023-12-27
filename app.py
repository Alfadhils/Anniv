from flask import Flask, render_template, request, send_file
from face_authenticator import FaceAuthenticator
from PIL import Image
import os

app = Flask(__name__)
authenticator = FaceAuthenticator(db_path='db')

# Initialize prediction as None
prediction = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=prediction)

@app.route('/', methods=['POST'])
def predict():
    global prediction
    imagefile = request.files['imagefile']
    image_path = os.path.join('images', imagefile.filename)
    imagefile.save(image_path)

    # Set prediction based on authenticator result (1 for true, 0 for false)
    prediction = authenticator.predict(image_path, plot=False)

    return render_template('index.html', prediction=prediction)

@app.route('/retry', methods=['GET'])
def retry():
    global prediction
    prediction = None
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False)
