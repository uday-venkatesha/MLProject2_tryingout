import numpy as np
import pandas as pd
import os

def generate_classification_data(n_samples=1000, n_features=10, random_state=42):
    """
    Generate synthetic data for binary classification.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        The number of samples.
    n_features : int, default=10
        The number of features.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    y : ndarray of shape (n_samples,)
        The target values (0 or 1).
    """
    np.random.seed(random_state)
    
    # Generate random feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Create a binary target with non-linear decision boundary
    # Use a combination of features to create a complex boundary
    y = np.zeros(n_samples)
    
    # First rule - quadratic relationship
    rule1 = (X[:, 0]**2 + X[:, 1]**2 < 4)
    
    # Second rule - interaction term
    rule2 = (X[:, 2] * X[:, 3] > 0)
    
    # Third rule - simple threshold
    rule3 = (X[:, 4] > 0)
    
    # Combine rules with some noise
    y = ((rule1 & rule2) | rule3).astype(int)
    
    # Add some noise (flip ~5% of labels)
    noise_mask = np.random.random(n_samples) < 0.05
    y[noise_mask] = 1 - y[noise_mask]
    
    return X, y

def generate_and_save_datasets():
    """Generate and save training, validation, and test datasets."""
    # Create datasets directory if it doesn't exist
    os.makedirs('BoostingTrees/tests', exist_ok=True)
    
    # Generate training data
    X_train, y_train = generate_classification_data(n_samples=1000, random_state=42)
    train_data = pd.DataFrame(np.column_stack([X_train, y_train]), 
                             columns=[f'feature_{i}' for i in range(X_train.shape[1])] + ['target'])
    train_data.to_csv('BoostingTrees/tests/train_data.csv', index=False)
    
    # Generate test data
    X_test, y_test = generate_classification_data(n_samples=300, random_state=123)
    test_data = pd.DataFrame(np.column_stack([X_test, y_test]), 
                            columns=[f'feature_{i}' for i in range(X_test.shape[1])] + ['target'])
    test_data.to_csv('BoostingTrees/tests/test_data.csv', index=False)
    
    # Generate a smaller test set for quick tests
    X_small, y_small = generate_classification_data(n_samples=50, random_state=456)
    small_data = pd.DataFrame(np.column_stack([X_small, y_small]), 
                             columns=[f'feature_{i}' for i in range(X_small.shape[1])] + ['target'])
    small_data.to_csv('BoostingTrees/tests/small_test.csv', index=False)
    
    print("Datasets generated and saved to BoostingTrees/tests/")

if __name__ == "__main__":
    generate_and_save_datasets()