import numpy as np
from flask import Flask, request, redirect, render_template
from tensorflow import keras
from keras.preprocessing import image

app = Flask(__name__)
model = keras.models.load_model('Covid_Vgg.h5')
def get_model():
    global model
    model=load_model("Covid_Vgg.h5")
    print("model loaded")
    
def preprocess_image(image,target_size):
    if image.mode !="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    
    return image

print("loading model")
get_model()
@app.route('/')
def home():
    
    return render_template('home.html')
    

@app.route('/predict',methods = ['GET','POST'])
def predict():
    message=request.get_json(force=True)
    encoded=str(message['image'])
    #print(encoded)
    encoded_pure=encoded.split(',')

    decoded=base64.b64decode(str(encoded_pure[1]))

    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(150,150))
    
    prediction=model.predict(processed_image).tolist()
    print(prediction)
    if prediction[0][0]<=0.5:
        covid_score=(1-prediction[0][0])*100
        normal_score=(prediction[0][0])*100
    else:
        covid_score=(1-prediction[0][0])*100
        normal_score=(prediction[0][0])*100
    
    response={
            "prediction":{
                    "covid":covid_score, 
                    "normal":normal_score
                    }
            }
    return str(covid_score)


if __name__ == '__main__':
    app.run(debug=True)
