"""
Train and evaluate a DecisionTreeRegressor model on the Boston Housing dataset.
"""

from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model, display_results


def main():
    """Main function to train and evaluate DecisionTreeRegressor."""
    print("Loading Boston Housing dataset...")
    df = load_data()
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    print("\nPreprocessing data...")
    # For DecisionTreeRegressor, we don't need to scale the features
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        df, 
        target_column='MEDV',
        test_size=0.2,
        random_state=42,
        scale=False  # Decision trees don't require scaling
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\nTraining DecisionTreeRegressor...")
    model = DecisionTreeRegressor(random_state=42)
    model = train_model(model, X_train, y_train)
    print("Training completed!")
    
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, X_test, y_test)
    
    # Display results
    display_results("DecisionTreeRegressor", results)


if __name__ == "__main__":
    main()
