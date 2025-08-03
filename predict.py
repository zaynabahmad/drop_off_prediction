import joblib
import numpy as np

# Load trained model
model = joblib.load('dropout_gb_model.pkl')

def get_user_input():
    print("\n Enter user behavior data:")
    try:
        session_freq = float(input("Average sessions per week (e.g., 2.5): "))
        time_spent = float(input("Average time spent per session in minutes (e.g., 45.3): "))
        progress_logs = float(input("Goal check-in rate (between 0 and 1, e.g., 0.4): "))

        # Input validation
        if not (0 <= progress_logs <= 1):
            raise ValueError("Check-in rate must be between 0 and 1.")

        return np.array([[session_freq, time_spent, progress_logs]])
    except ValueError as e:
        print(f" Invalid input: {e}")
        return None

def predict_dropout(input_features):
    prediction = model.predict(input_features)[0]
    risk_score = model.predict_proba(input_features)[0][1]  # Probability of dropout (class 1)

    print("\n Prediction Result:")
    if prediction == 1:
        print(" Prediction: The user is likely to DROP OUT.")
    else:
        print(" Prediction: The user is likely to STAY.")

    print(f" Risk Score (Probability of dropout): {risk_score:.2f}")

if __name__ == "__main__":
    user_features = get_user_input()
    if user_features is not None:
        predict_dropout(user_features)
