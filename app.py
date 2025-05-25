import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Salary Prediction Based on Experience and Age")

data = pd.read_csv("salary_data.csv")
st.write(data)

x = data[['Experience_in_Years', 'Age']]
y = data['Salary_in_Thousands']

model = LinearRegression()
model.fit(x, y)

experience = st.number_input("Enter Experience in Years:", 0.0, 20.0, step=1.0)
age = st.number_input("Enter Age:", 18.0, 60.0, step=1.0)

prediction = model.predict([[experience, age]])
st.write("Predicted Salary (in Thousands):", prediction[0])