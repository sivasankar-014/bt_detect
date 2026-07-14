from flask import Flask,render_template,request
import tensorflow as tf
import numpy as np
from PIL import Image

app=Flask(__name__)
model=tf.keras.models.load_model('brain_tumor_model.keras')

classes = ["glioma", "no_tumor"]

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image)
    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    
    file=request.files['image']
    
    image=Image.open(file)
    processed_image=preprocess_image(image)
    
    prediction=model.predict(processed_image)
    
    class_index=np.argmax(prediction)
    
    result=classes[class_index]
    
    confidence=np.max(prediction)*100
    
    return render_template('index.html',
                           prediction=result,
                           confidence=f'{confidence:.2f}')
if __name__ == '__main__':
    app.run(debug=True)
        
    
    














