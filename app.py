import numpy as np
from flask import Flask, request, redirect, render_template
from tensorflow import keras
from keras.preprocessing import image

app = Flask(__name__)
model = keras.models.load_model('Covid_Vgg.h5')


@app.route('/')
def home():
    
    return render_template('home.html')
    

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == "POST":
        if request.files:
            
            
            int_features = request.files["image_file"]
            print(int_features)
            process_features=image.load_img(int_features, target_size=(150, 150))
            final_features = np.expand_dims(process_features, axis=0)
            output = model.predict_classes(final_features)
            print(output)
            for x in output:
                if x==0:
                    prediction="Patient has Covid +ve"
                    print(prediction)
                else:
                    prediction="No Covid Symptoms Detected"
                    print(prediction)
            
            return redirect(request.url)
    
    
    return render_template('home.html', prediction_text="Result is {}".format(prediction))
    


if __name__ == '__main__':
    app.run(debug=True)
