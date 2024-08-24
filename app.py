from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    # Combine into a feature array
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    final_features = [np.array(features)]

    # Predict using the loaded model
    prediction = model.predict(final_features)

    result = "Positive" if prediction[0] else "Negative"

    # Define remarks and suggestions based on prediction
    if result == "Positive":
        remarks = "The model predicts a higher likelihood of diabetes. Please consult a healthcare provider for further diagnosis."
        suggestion = "Consider a healthier diet, regular exercise, and regular monitoring of your blood sugar levels."
    else:
        remarks = "The model predicts a lower likelihood of diabetes. Continue maintaining a healthy lifestyle."
        suggestion = "Keep up with regular health check-ups, and maintain a balanced diet and exercise routine."

    # Render the result page with the prediction and suggestions
    return render_template('result.html', prediction_text=f'Diabetes Prediction: {result}', remarks=remarks,
                           suggestion=suggestion)


if __name__ == "__main__":
    app.run(debug=True)
