from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("heart-1"
                           ".html")


@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    prediction=model.predict(final)


    if int(prediction) == 1:
        return render_template('heart-1.html', pred="You are safe....")
    else:
        return render_template('heart-1.html', pred="Heart is in danger!!! check out for help - www.hdfhd.com")


if __name__ == '__main__':
    app.run(debug=True)