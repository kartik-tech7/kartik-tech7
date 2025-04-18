import numpy as np

def predict_water_quality(model, input_data):
    prediction = model.predict([input_data])
    return "Safe to Drink" if prediction[0] == 1 else "Not Safe to Drink"
