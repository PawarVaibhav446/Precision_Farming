from email import message
from unicodedata import name
from flask import Flask, redirect, render_template, request, url_for, session, send_file
import uuid as uuid
import requests
import config
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier







def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None
    
app = Flask(__name__)
app.secret_key = 'yash'
upload_folder = 'static/images/pics/'
app.config['upload_folder'] = upload_folder
def sessionblank():
    session["successful"] = ''



@app.route('/')
def index():
    sessionblank()
    return render_template("index.html")



@app.route('/aboutus')
def aboutus():
    sessionblank()
    return render_template("dtree.html")


@app.route('/knnpage')
def knn():
    sessionblank()
    return render_template("knn.html")








@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        df= pd.read_csv('Crop_recommendation.csv')
        labels = df.pop("label")
        x_train, x_test, y_train, y_test = train_test_split(df,labels,test_size=0.25)
        rf= RandomForestClassifier()
        rf.fit(x_train,y_train)
        demo= rf.score(x_test,y_test)
        y_predicted = rf.predict(x_test)
       



        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = rf.predict(data)
            final_prediction = my_prediction[0]

            

            cm = confusion_matrix(y_test,y_predicted)
            plt.figure(figsize=(10,7))
            print(demo)
            plt.xlabel("predicted")
            plt.ylabel("Truth")
            sn.heatmap(cm, annot=True)

            return render_template('crop-result.html', prediction=final_prediction,demo=demo, matrix=cm, title=title)

        else:

            return render_template('try_again.html', title=title)



@ app.route('/dtree', methods=['POST'])
def crop_prediction_tree():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])


        df= pd.read_csv('Crop_recommendation.csv')
        labels = df.pop("label")
        x_train, x_test, y_train, y_test = train_test_split(df,labels,test_size=0.25)
        dtree = tree.DecisionTreeClassifier()
        dtree.fit(x_train,y_train)
        demo= dtree.score(x_test,y_test)
        y_dtpred = dtree.predict(x_test)
       




        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = dtree.predict(data)
            final_prediction = my_prediction[0]
            

            cm = confusion_matrix(y_test,y_dtpred)
            plt.figure(figsize=(10,7))
            print(demo)
            sn.heatmap(cm, annot=True)
            plt.xlabel("predicted")
            plt.ylabel("Truth")

            return render_template('crop-result.html', prediction=final_prediction, demo=demo, matrix=cm, title=title)

        else:

            return render_template('try_again.html', title=title)
        
@ app.route('/knn', methods=['POST'])
def crop_prediction_knn():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        df= pd.read_csv('Crop_recommendation.csv')
        labels = df.pop("label")
        x_train, x_test, y_train, y_test = train_test_split(df,labels,test_size=0.25)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train,y_train)
        demo= knn.score(x_test,y_test)
        y_predicted = knn.predict(x_test)
       




        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = knn.predict(data)
            final_prediction = my_prediction[0]
            

            cm = confusion_matrix(y_test,y_predicted)
            plt.figure(figsize=(10,7))
            print(demo)
            sn.heatmap(cm, annot=True)
            plt.xlabel("predicted")
            plt.ylabel("Truth")

            return render_template('crop-result.html', prediction=final_prediction, matrix=cm,demo=demo,title=title)

        else:

            return render_template('try_again.html', title=title)
        

if(__name__) == '__main__':
    app.run(debug=True)
