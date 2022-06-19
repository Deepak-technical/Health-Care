
from flask import Flask,render_template, url_for ,flash , redirect
import joblib
from flask import request
import numpy as np
import os
import pickle
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
app=Flask(__name__)



# Deep Learning Models
model = load_model('models/malaria_detector.h5')
# model222=load_model("models/my_model.h5")



# Importing ALl Pickle Files
Diabetics_Model = pickle.load(open('models/Diabetics.pkl', 'rb'))
Kideny_Model = pickle.load(open('models/Kidney1.pkl', 'rb'))
Liver_Model = pickle.load(open('models/liver.pkl', 'rb'))
Heart_model = pickle.load(open('models/Heart.pkl', 'rb'))


# Byuilding APIs For Deep learning
def Maleria_api(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224,3))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    preds = model.predict(x)
    print(preds)
    if preds==[[1.]]:
        preds="Negative Report Maleria Infected  Cell Has  Not detected"
        color="green"
    else:
        preds=" Positive Report Maleria Infected Cell Found  in your sample specium "
        color="red"

    return preds,color





MODEL_PATH = 'models/Detection_Covid_19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict(xtest_image)
    return preds


# Maleria Prediction Model On Submit


@app.route('/malaria', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds,color = Maleria_api(file_path, model)
        result=preds
        return render_template("Maleria.html",result=result,color=color)
    else:
        return render_template("Maleria.html")


@app.route('/covid', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        if preds[0][0] == 0:
            prediction = 'Sorry You Have Been Tested Positive  For Covid-19 Virus Please Take Necessary Preventaions And Isolate Immediately'
            color="red"
        else:
            prediction = 'Good You have Tested Negative for Covid-19,Make sure that You should Take care All Preventions'
            color="green"
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1)               # Convert to string
        return render_template("Covid.html",result=prediction,color=color)
    else:
        return render_template("Covid.html")





@app.route("/")
def home():
    return render_template("home.html")
 


@app.route("/diabetes",methods=["POST","GET"])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])


        data=np.array([[preg,glucose,bp,st,insulin,bmi,dpf,age]])
        prediction=Diabetics_Model.predict(data)

        if(prediction==1):
            prediction="Opps! It Seems That You Have a High Risk Of Getting  Diabetics"
            color="red"
        else:
            prediction="Great! You Don't Have Diabetics Please Maintain a Good diet For Healthy Life"
            color="green"

        return render_template ("diabetes.html",prediction_text=prediction,color=color)

    else:
        return render_template("diabetes.html")
    

@app.route("/heart",methods=["POST","GET"])
def heart():
    if request.method=="POST":
        age = int(request.form['age'])
        Gender = int(request.form['sex'])
        chestpaintype = int(request.form['chest pain type'])
        trestbps = int(request.form['trestbps'])
        serum = int(request.form['serum cholestoral in mg/dl'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = int(request.form['oldpeak'])
        slope = int(request.form['slope'])
        thal = int(request.form['thal'])
        fbs= int(request.form['fbs'])
        ca= int(request.form['ca'])

        data=np.array([[age,Gender,chestpaintype,trestbps,serum,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        prediction=Heart_model.predict(data)
        if(prediction==1):
            prediction="Opps! It Seems You Have Some Heart Diseses Like  Heart valve disease ,Disease of the heart muscle, Coronary artery disease"
            color="red"
        else:
            prediction="Great! You Have an Healthy Heart"
            color="green"
        return render_template ("heart.html",prediction=prediction, color=color)

    else:    
       return render_template("heart.html")


@app.route("/liver",methods=["POST","GET"])
def liver():
    if request.method=="POST":
        age = int(request.form['age'])
        Gender = int(request.form['Gender'])
        Total_Bilirubin = int(request.form['Total_Bilirubin'])
        Direct_Bilirubin = int(request.form['Direct_Bilirubin'])
        Alamine_Aminotransferase= int(request.form['Alamine_Aminotransferase'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = int(request.form['Total_Protiens'])
        Albumin = int(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = int(request.form['Albumin_and_Globulin_Ratio'])
        data=np.array([[age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender]])   
        prediction=Liver_Model.predict(data) 
        if(prediction==1):
            prediction="Opps!! It Seems That Your Liver have Infected by some  Diseases. Consult To A Doctor Immediately"
            color="red"
        else:
            prediction="Great! You Don't have a Any Tyoes Liver And Take Care"
            color="green"
        return render_template("liver.html",prediction=prediction,color=color)
    else:
        return render_template("liver.html")

    
    # return render_template("liver.html")

@app.route("/kidney",methods=["POST","GET"])
def kidney():
    if request.method=="POST":
        age = int(request.form['age'])
        bp = int(request.form['bp'])
        al = int(request.form['al'])
        pcc = int(request.form['pcc'])
        bgr = int(request.form['bgr'])
        bu = int(request.form['bu'])
        sc = int(request.form['sc'])
        hemo = int(request.form['hemo'])
        pcv = int(request.form['pcv'])
        htn = int(request.form['htn'])
        dm = int(request.form['dm'])
        appet = int(request.form['appet'])
        data=np.array([[age,bp,al,pcc,bgr,bu,sc,hemo,pcv,htn,dm,appet]])
        prediction=Kideny_Model.predict(data)
        if(prediction==1):
            prediction="Opps! It Seems That You Have Chronic Kidney Diseses (CKD)"
            color="red"
        else:
            prediction="Great! You Don't have Kidney diseases"
            color="green"
        return render_template("kidney.html",prediction=prediction,color=color)
    else:
        return render_template("kidney.html")






if __name__ == "__main__":
    app.run(debug=True)
