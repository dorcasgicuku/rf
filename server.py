import numpy as np
from flask import Flask, request, jsonify
import pickle
import flask
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('random_Forest_model.pkl','rb'))


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        day=flask.request.form['day']
        hour=flask.request.form['hour']
        minute=flask.request.form['minute']
        dni=flask.request.form['dni']
        dhi=flask.request.form['dhi']
        cloud=flask.request.form['cloud_type']
        wind=flask.request.form['wind_speed']
        temperature = flask.request.form['temp']
        humidity = flask.request.form['humidity']
        pressure = flask.request.form['pressure']
        
        input_variables = pd.DataFrame([[dni,dhi,cloud,wind,temperature, humidity, pressure,day,hour,minute]],
                                       columns=['DNI','DHI','Cloud Type','Wind Speed','Temperature', 'Relative Humidity','Pressure','Day','Hour','Minute' ],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',result=prediction)

if __name__ == '__main__':
    app.run()
