from flask import Flask, render_template, request 
import numpy as np
app=Flask(__name__) 
import pickle

model=pickle.load(open('ar_rfc.pkl','rb'))
ss1=pickle.load(open('ar_ss.pkl','rb'))
le1=pickle.load(open('le1.pkl','rb'))
le2=pickle.load(open('le2.pkl','rb'))
le3=pickle.load(open('le3.pkl','rb'))
le4=pickle.load(open('le4.pkl','rb'))
le5=pickle.load(open('le5.pkl','rb'))
le6=pickle.load(open('le6.pkl','rb'))
le7=pickle.load(open('le7.pkl','rb'))
le8=pickle.load(open('le8.pkl','rb'))
le9=pickle.load(open('le9.pkl','rb'))
le10=pickle.load(open('le10.pkl','rb'))

@app.route('/')
def home():
        return render_template('Home.html')
@app.route('/review')
def review():
    return render_template('review.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pred', methods=['POST'])
def predict():
        Airline_name=request.form['Airline name']
        Seat_Type=request.form['Seat Type']
        Type_Of_Traveller=request.form['Type Of Traveller']
        Origin=request.form['Origin']
        Destination=request.form['Destination']
        Month_Flown=request.form['Month Flown']
        Year_Flown=request.form['Year Flown']
        Verified=request.form['Verified']
        S_C=int(request.form['S_C'])
        F_B=int(request.form['F_B'])
        G_S=int(request.form['G_S'])
        O_R=int(request.form['O_R'])
        data=[[Airline_name, Seat_Type, Type_Of_Traveller, Origin, Destination,
                Month_Flown,Year_Flown, Verified,S_C,F_B,G_S,O_R]]
        encoded_data = [
            le1.transform([Airline_name])[0],
            le2.transform([Seat_Type])[0],
            le3.transform([Type_Of_Traveller])[0], 
            le4.transform([Origin])[0], 
            le5.transform([Destination])[0], 
            le6.transform([Month_Flown])[0], 
            le7.transform([Year_Flown])[0], 
            le8.transform([Verified])[0],
            [S_C][0],[F_B][0],[G_S][0],
            le9.transform([O_R])[0]
        ]

        print (encoded_data)
        prediction=model.predict(ss1.transform([encoded_data]))
        if prediction== 1:
           a="Recommended"
           return render_template('review.html',result=a)
        else:
          b="Not Recommended"
          return render_template('review.html',result=b)
        
if __name__ == "__main__":
    app.run(debug=True)