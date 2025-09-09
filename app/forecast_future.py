import numpy as np
from tensorflow.keras.models import load_model
import joblib

def forecast_future(model_path, scaler_path, recent_data, days_to_predict=30):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    last_60 = recent_data[-60:].reshape(-1, 1)
    scaled_last_60 = scaler.transform(last_60)
    input_seq = list(scaled_last_60)

    predictions = []

    for _ in range(days_to_predict):
        x_input = np.array(input_seq[-60:]).reshape(1, 60, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0][0]
        input_seq.append([pred_scaled])
        predictions.append(pred_scaled)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return forecast
