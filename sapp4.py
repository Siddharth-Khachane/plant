from flask import Flask, render_template, request
import numpy as np
from keras.models import model_from_yaml
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model=load_model('pnp_2.h5')
model1=load_model('modelv2_1.h5')
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
#img_size = 100

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def resultForm():
    
    global COUNT
    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))    
    image1 = cv2.imread('static/{}.jpg'.format(COUNT))
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(img1,(50,50))
    reshaped=np.reshape(resized,(1,50,50,1))
    #data[0]=img
    
    pred = model.predict(reshaped)
    a=pred[0]
    np.ndarray.tolist(a)
    ma=max(a)
    str=""
    for i in range(len(a)):
        if a[i]==ma:
            pos=i
    if pos==0:
        str="Not a leaf"
    else:
        img2 = image.load_img('static/0.jpg', target_size=(128,128))
        img_array = image.img_to_array(img2)
        img_batch = np.expand_dims(img_array, axis=0)
        pred = model1.predict(img_batch)
        if np.argmax(pred) == 0:
            str="Healthy"
        elif np.argmax(pred) == 1:
            str="Multiple Diseases"
        elif np.argmax(pred) == 2:
            str="Rust"
        else: 
            str="Scab"
    return render_template('result.html', data=str)

if __name__ == "__main__":
    app.run(port=3000,debug=True)
