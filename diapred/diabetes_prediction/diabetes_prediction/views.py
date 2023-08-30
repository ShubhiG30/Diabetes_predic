from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    data = pd.read_csv(r'C:\Users\91752\Downloads\archive (3)\diabetes.csv')
    diabetes_df_copy = data.copy(deep=True)
    diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace=True)
    diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace=True)
    diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
    diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace=True)
    diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)
    sc_X = StandardScaler()
    X = pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"], axis=1), ), columns=['Pregnancies',
                                                                                                'Glucose',
                                                                                                'BloodPressure',
                                                                                                'SkinThickness',
                                                                                                'Insulin', 'BMI',
                                                                                                'DiabetesPedigreeFunction',
                                                                                                'Age'])
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = rfc.predict([[val1, val2, val3, val4, val5, val6, val7, val8 ]])

    result1 = ""
    if pred==[1]:
        result1 = "Positive"
    else:
        result1 = "Negative"
    return render(request,"predict.html", {"result2":result1})
