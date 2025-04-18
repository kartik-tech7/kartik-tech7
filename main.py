from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model
from prediction import predict_water_quality

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('water_potability.csv')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Make prediction (example)
    sample_data = [7.0, 204.9, 20791.32, 368.52, 564.3, 10.38, 4.5, 396.4, 10.0]
    result = predict_water_quality(model, sample_data)
    print("\nPrediction for sample input:", result)
