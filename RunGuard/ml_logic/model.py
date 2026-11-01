import xgboost as xgb


def xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model on the provided training data and evaluate it on the test data.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.

    Returns:
    None
    """


    # 2. Compute imbalance ratio for class weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_count / pos_count



    # 3. Initialize XGBoost with a fixed seed and class weight balancing
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight_val,
        random_state=42,  # Fixed seed for reproducibility
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # 4. Train the model (No hyperparameter tuning)
    model.fit(X_train, y_train)

    # 5. Score / predict probabilities on the test table
    y_scores = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate using the previously defined plot_roc_pr_curves function
    #plot_roc_pr_curves(y_test, y_scores)
### put here everything for the model: initialize, compile, train, evaliate, predict
### save function for trained model in the cloud (ml flow?)
### load function to load the model later in the api
