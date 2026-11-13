from RunGuard.ml_logic import model
from RunGuard.ml_logic import train_test_data
import pandas as pd

def main():
    """
    Main function to initialize and train the XGBoost model.
    """

    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")

    # Load training and test data (test = held-out athletes)
    X_train, y_train, X_test, y_test = train_test_data.train_test_data_prep(df)

    # Initialize the XGBoost model
    xgb_model = model.initialize_xgboost_model(y_train)

    # Train the model
    _, trained_model = model.train_model(xgb_model, X_train, y_train)

    # Evaluate on the test athletes against the no-skill baseline
    print("Test set evaluation:")
    print(model.evaluate_model(trained_model, X_test, y_test))

    # Save the trained model
    model.save_model(trained_model)

    print("Model training and saving completed successfully.")


if __name__ == "__main__":
    main()
