from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)
@app.route('/')

def front_page():
    return render_template('frontpage.html')
       #project-id_PNT2022TMID53213

@app.route('/hearttt',methods=['POST'])
def heartt():
    
        age = request.form['age']
        sex = request.form['sex']
        chest = request.form['chest']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        #project-id_PNT2022TMID53213
       
        model2 = pickle.load(open('./static/heart_model.pkl','rb'))
        input_data = [age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        for i in range(len(input_data)):
            input_data[i]=float(input_data[i])
        print(input_data)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model2.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 0):
            print("According to the given details person does not have Heart Disease")
            senddata='According to the given details person does not have Heart Disease'
            print(senddata)
        else:
            print("According to the given details person does not have Heart Disease")
            senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
            
        return render_template('result.html',resultvalue=senddata)
if __name__ == '__main__':
    app.run()