"""
Train and evaluate a KernelRidge model on the Boston Housing dataset.
"""

from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model, display_results


def main():
    """Main function to train and evaluate KernelRidge."""
    print("Loading Boston Housing dataset...")
    df = load_data()
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    print("\nPreprocessing data...")
    # For KernelRidge, scaling is recommended for better performance
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, 
        target_column='MEDV',
        test_size=0.2,
        random_state=42,
        scale=True  # KernelRidge benefits from scaling
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\nTraining KernelRidge...")
    # Using RBF kernel with default parameters
    model = KernelRidge(alpha=1.0, kernel='rbf')
    model = train_model(model, X_train, y_train)
    print("Training completed!")
    
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, X_test, y_test)
    
    # Display results
    display_results("KernelRidge", results)


if __name__ == "__main__":
    main()
