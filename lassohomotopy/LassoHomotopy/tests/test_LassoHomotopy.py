import csv
import time
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel 
import matplotlib.pyplot as plt
import pytest
import warnings
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

def load_data():
    data = []
    with open("./collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    X = np.array([[float(v) for k, v in row.items() if k.startswith('X')] for row in data])
    y = np.array([float(row['target']) for row in data])
    return X, y.reshape(-1)

def test_basic_prediction():
    """Comprehensive test of basic prediction 
    functionality with detailed reporting with report"""
    # Initialize model with default parameters
    model = LassoHomotopyModel()
    X, y = load_data()
  
    # Fit model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    coefficients = results.coef_
    
    # Calculate performance metrics
    mse = np.mean((y - preds)**2)
    mae = np.mean(np.abs(y - preds))
    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
    corr = np.corrcoef(y, preds)[0, 1]
    
    # Sparsity analysis
    num_non_zero = np.sum(np.abs(coefficients) > model.model.tol)
    sparsity = 1 - (num_non_zero / len(coefficients))
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    BASIC PREDICTION TEST REPORT
    {'='*60}
    Data Characteristics:
    - Samples: {X.shape[0]}
    - Features: {X.shape[1]}
    - Target mean: {np.mean(y):.4f}
    - Target std: {np.std(y):.4f}
    
    Model Parameters:
    - lambda (λ): {model.lambda_par:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    - Intercept: {results.intercept_:.4f}
    
    Performance Metrics:
    - R-squared: {r2:.4f}
    - Mean Squared Error: {mse:.4f}
    - Mean Absolute Error: {mae:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Sparsity Analysis:
    - Non-zero coefficients: {num_non_zero}/{len(coefficients)}
    - Sparsity ratio: {sparsity:.1%}
    - Largest coefficient: {np.max(np.abs(coefficients)):.4f}
    
    Prediction Statistics:
    - Min prediction: {np.min(preds):.4f}
    - Max prediction: {np.max(preds):.4f}
    - Prediction range: {np.max(preds) - np.min(preds):.4f}
    {'='*60}
    """
    print(report)
    plt.style.use('dark_background')

    # Create figure with custom title
    fig = plt.figure(figsize=(18, 6), facecolor='#121212')
    fig.suptitle('Basic Predictions', 
                fontsize=14, fontweight='bold', y=1.02)

    # Plot 1: Actual vs Predicted values
    ax1 = plt.subplot(1, 3, 1)
    sc = ax1.scatter(y, preds, alpha=0.7, color='#1f77b4', 
                    edgecolors='white', linewidths=0.3)
    ax1.plot([min(y), max(y)], [min(y), max(y)], 
            'r--', linewidth=1.5, label='Perfect Prediction')
    ax1.set_title('Actual vs Predicted Values', fontsize=13, pad=15, color='white')
    ax1.set_xlabel('Actual Values', fontsize=11, color='white')
    ax1.set_ylabel('Predicted Values', fontsize=11, color='white')
    ax1.grid(color='gray', linestyle=':', alpha=0.2)
    ax1.legend(fontsize=10)

    # Plot 2: Coefficient magnitudes (enhanced stem plot)
    ax2 = plt.subplot(1, 3, 2)
    markerline, stemlines, baseline = ax2.stem(
        np.arange(len(coefficients)), 
        coefficients,
        linefmt='#2ca02c',
        markerfmt='o',
        basefmt='gray'
    )
    plt.setp(stemlines, 'linewidth', 1.5)
    plt.setp(markerline, 'markersize', 5, 'color', '#ff7f0e')
    ax2.axhline(0, color='white', linewidth=0.8, alpha=0.5)
    ax2.set_title('Feature Coefficients', fontsize=13, pad=15, color='white')
    ax2.set_xlabel('Feature Index', fontsize=11, color='white')
    ax2.set_ylabel('Coefficient Value', fontsize=11, color='white')
    ax2.grid(color='gray', linestyle=':', alpha=0.2)

    # Highlight significant coefficients
    sig_coefs = np.where(np.abs(coefficients) > 0.1)[0]
    for i in sig_coefs:
        ax2.annotate(f'{coefficients[i]:.2f}',
                    (i, coefficients[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    color='white')

    # Plot 3: Prediction error distribution (enhanced)
    ax3 = plt.subplot(1, 3, 3)
    errors = y - preds
    n, bins, patches = ax3.hist(errors, bins=30, 
                            color='#9467bd', 
                            alpha=0.8,
                            edgecolor='white',
                            linewidth=0.5)
    ax3.axvline(0, color='white', linestyle='--', linewidth=1.2)
    ax3.axvline(np.mean(errors), color='red', linestyle='-', 
            linewidth=1.5, label=f'Mean: {np.mean(errors):.2f}')
    ax3.set_title('Prediction Error Distribution', fontsize=13, pad=15, color='white')
    ax3.set_xlabel('Prediction Error', fontsize=11, color='white')
    ax3.set_ylabel('Frequency', fontsize=11, color='white')
    ax3.legend(fontsize=10)
    ax3.grid(color='gray', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig('../images/basic_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
        
    # Assertions with helpful messages
    assert preds is not None, "Model failed to generate predictions"
    assert preds.shape == y.shape, (
        f"Prediction shape mismatch. Expected {y.shape}, got {preds.shape}"
    )
    assert not np.allclose(preds, 0), (
        "All predictions are zero - model may have failed to learn"
    )
    assert r2 > 0.3, (
        f"Low R-squared value ({r2:.4f}) - model may not be fitting well"
    )
    assert sparsity >= 0.1, (
        f"Low sparsity ({sparsity:.1%}) - consider increasing lambda_par"
    )

def test_prediction_visualization():
    """Generate comprehensive prediction visualization report"""
    # Initialize model with default parameters
    model = LassoHomotopyModel(lambda_par=0.1, fit_intercept=True, max_iter=1000)
    X, y = load_data()
    
    # Standardize features for better performance
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    
    # Calculate performance metrics
    mse = np.mean((y - preds)**2)
    mae = np.mean(np.abs(y - preds))
    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
    corr = np.corrcoef(y, preds)[0, 1]
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    PREDICTION VISUALIZATION REPORT
    {'='*60}
    Model Performance Metrics:
    - Mean Squared Error: {mse:.4f}
    - Mean Absolute Error: {mae:.4f}
    - R-squared: {r2:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Prediction Statistics:
    - Actual mean: {np.mean(y):.4f} ± {np.std(y):.4f}
    - Predicted mean: {np.mean(preds):.4f} ± {np.std(preds):.4f}
    - Min prediction: {np.min(preds):.4f}
    - Max prediction: {np.max(preds):.4f}
    
    Error Distribution:
    - Median absolute error: {np.median(np.abs(y - preds)):.4f}
    - 90th percentile error: {np.percentile(np.abs(y - preds), 90):.4f}
    - Max absolute error: {np.max(np.abs(y - preds)):.4f}
    {'='*60}
    """
    print(report)
    
    # Create enhanced visualizations
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Actual vs Predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y, preds, alpha=0.6, color='royalblue')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=1)
    plt.title('Actual vs Predicted Values', fontsize=12)
    plt.xlabel('Actual Values', fontsize=10)
    plt.ylabel('Predicted Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Index-based comparison
    plt.subplot(1, 3, 2)
    sample_indices = np.arange(len(y))
    plt.scatter(sample_indices, y, color='blue', label='Actual', alpha=0.6, s=20)
    plt.scatter(sample_indices, preds, color='red', label='Predicted', alpha=0.6, s=20)
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title('Sample-wise Comparison', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(1, 3, 3)
    errors = y - preds
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution', fontsize=12)
    plt.xlabel('Prediction Error', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save and show plot
    plt.savefig('../images/prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional diagnostic: Residuals vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, errors, alpha=0.6, color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=12)
    plt.xlabel('Predicted Values', fontsize=10)
    plt.ylabel('Residuals', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('../images/ResidualsvsPredicted.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_empty_input():
    """Test with empty input arrays"""
    model = LassoHomotopyModel()
    X = np.array([])
    y = np.array([])
    
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_single_feature():
    """Test LassoHomotopy with single feature input (with visualization)"""
    import matplotlib.pyplot as plt
    
    # Initialize model with default parameters
    model = LassoHomotopyModel(lambda_par=0.1, fit_intercept=True, max_iter=1000)
    
    try:
        # Load and validate data
        X, y = load_data()
        assert X.size > 0, "X is empty"
        assert y.size > 0, "y is empty"
        
        # Select single feature and standardize
        X_single = X[:, 0:1]  # Maintain 2D shape (n_samples, 1)
        X_single = (X_single - np.mean(X_single)) / np.std(X_single)
        
        # Calculate feature-target correlation
        ft_corr = np.corrcoef(X_single.flatten(), y)[0,1]
        
        # Fit model
        results = model.fit(X_single, y)
        coefficients = results.coef_
        intercept = results.intercept_ if hasattr(results, 'intercept_') else 0
        coef_value = coefficients[0] if coefficients.size > 0 else 0
        
        # Generate predictions
        preds = results.predict(X_single)
        
        # Calculate performance metrics
        mse = np.mean((preds - y)**2)
        corr = np.corrcoef(preds, y)[0, 1]
        r_squared = 1 - (np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2))

        # Create visualization
        
        fig = plt.figure(figsize=(12, 6))
        
        fig.suptitle('Lasso Homotopy - Single Feature Analysis', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Scatter plot of actual vs predicted
        ax1 = plt.subplot(1, 2, 1)
        ax1.scatter(X_single, y, alpha=0.5, label='Actual', color='cyan')
        ax1.scatter(X_single, preds, alpha=0.5, label='Predicted', color='magenta')
        ax1.set_xlabel('Standardized Feature Value', color='white')
        ax1.set_ylabel('Target Value', color='white')
        ax1.set_title('Actual vs Predicted Values', color='white', pad=20)
        ax1.legend()
        ax1.grid(color='gray', linestyle=':', alpha=0.3)
        
        # Regression line plot
        ax2 = plt.subplot(1, 2, 2)
        x_vals = np.linspace(X_single.min(), X_single.max(), 100)
        y_vals = coef_value * x_vals + intercept
        ax2.scatter(X_single, y, alpha=0.3, color='cyan')
        ax2.plot(x_vals, y_vals, color='yellow', 
                linewidth=2,
                label=f'y = {coef_value:.2f}x + {intercept:.2f}')
        ax2.set_xlabel('Standardized Feature Value', color='white')
        ax2.set_ylabel('Target Value', color='white')
        ax2.set_title('Regression Line Fit', color='white', pad=20)
        ax2.legend()
        ax2.grid(color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        
        
        # Generate comprehensive report
        report = f"""
        {'='*60}
        SINGLE FEATURE TEST REPORT
        {'='*60}
        Data Characteristics:
        - Samples: {X_single.shape[0]}
        - Features: {X_single.shape[1]}
        - Target mean: {np.mean(y):.4f}
        - Target std: {np.std(y):.4f}
        - Feature-target correlation: {ft_corr:.4f}
        
        Model Parameters:
        - lambda_par (λ): {model.lambda_par:.4f}
        - Fit intercept: {model.fit_intercept}
        - Intercept value: {intercept:.4f}
        - Coefficient value: {coef_value:.4f}
        
        Performance Metrics:
        - Mean Squared Error: {mse:.4f}
        - R-squared: {r_squared:.4f}
        - Prediction-Target Correlation: {corr:.4f}
        
        {'='*60}
        """
        print(report)
        
        # Save and show plot
        plt.savefig('../images/single_feature_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
    
    
def test_high_dimensional_data():
    """Test LassoHomotopy on high-dimensional data (p > n) with detailed reporting"""
    # Initialize model with default parameters
    model = LassoHomotopyModel(lambda_par=0.1, fit_intercept=True, max_iter=10000)
    np.random.seed(42)
    
    # Generate high-dimensional data (10 samples, 20 features)
    X = np.random.randn(10, 20)
    y = np.random.randn(10)
    
    # Standardize features (important for Lasso)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model and make predictions
    results = model.fit(X, y)
    coefficients = results.coef_
    preds = results.predict(X)
    
    # Calculate performance metrics
    mse = np.mean((preds - y)**2)
    corr = np.corrcoef(preds, y)[0, 1]
    
    # Sparsity analysis
    num_non_zero = np.sum(np.abs(coefficients) > model.model.tol)
    sparsity = 1 - (num_non_zero / len(coefficients))
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    HIGH-DIMENSIONAL DATA TEST REPORT (n={X.shape[0]}, p={X.shape[1]})
    {'='*60}
    Model Parameters:
    - lambda_par (λ): {model.lambda_par:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    
    Performance Metrics:
    - Mean Squared Error: {mse:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Sparsity Analysis:
    - Total features: {len(coefficients)}
    - Non-zero coefficients: {num_non_zero} ({num_non_zero/len(coefficients):.1%})
    - Zero coefficients: {len(coefficients) - num_non_zero} ({sparsity:.1%})
    - Effective dimensionality reduction: {num_non_zero} →  {len(coefficients)-X.shape[0]}features
    
    Coefficient Statistics:
    - Largest absolute coefficient: {np.max(np.abs(coefficients)):.4f}
    - Smallest non-zero coefficient: {np.min(np.abs(coefficients[np.abs(coefficients) > model.model.tol])):.4f}
    - L1 norm of coefficients: {np.sum(np.abs(coefficients)):.4f}
    
    {'='*60}
    """
    print(report)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    plt.suptitle("High dimentional data",fontsize=14, fontweight='bold', y=1.02)
    # Plot coefficients
    plt.subplot(1, 3, 1)
    plt.stem(np.arange(len(coefficients)), coefficients, markerfmt=' ')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Feature Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y, preds)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot sorted absolute coefficients
    plt.subplot(1, 3, 3)
    sorted_abs = np.sort(np.abs(coefficients))[::-1]
    plt.plot(sorted_abs, 'o-')
    plt.axhline(model.model.tol, color='red', linestyle='--', label=f'Tolerance ({model.model.tol:.1e})')
    plt.title('Sorted Absolute Coefficients (log scale)')
    plt.xlabel('Coefficient Rank')
    plt.ylabel('Absolute Value')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../images/HighdimentionalData.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Assertions with helpful messages
    assert preds is not None, "Model failed to generate predictions"
    assert preds.shape == y.shape, (
        f"Prediction shape mismatch. Expected {y.shape}, got {preds.shape}"
    )
    assert sparsity > 0.05, (
        f"Insufficient sparsity (got {sparsity:.1%}, expected >30%).\n"
        f"Try increasing lambda_par (current: {model.lambda_par}) or checking data scaling."
    )
    

def test_sparse_solution():
    """Test that solution shows reasonable sparsity with detailed reporting"""
    # Initialize model with high lambda_par to encourage sparsity
    model = LassoHomotopyModel(lambda_par=1.0, fit_intercept=True, max_iter=10000)
    X, y = load_data()
    
    # Fit model
    results = model.fit(X, y)
    coefficients = results.coef_
    
    # Calculate sparsity metrics
    non_zero_mask = np.abs(coefficients) > model.model.tol
    num_non_zero = np.sum(non_zero_mask)
    num_zero = len(coefficients) - num_non_zero
    sparsity = num_zero / len(coefficients)
    
    # Get coefficient statistics
    non_zero_coeffs = coefficients[non_zero_mask]
    coeff_stats = {
        'max': np.max(np.abs(coefficients)),
        'min_nonzero': np.min(np.abs(non_zero_coeffs)) if num_non_zero > 0 else 0,
        'mean_nonzero': np.mean(np.abs(non_zero_coeffs)) if num_non_zero > 0 else 0,
        'std_nonzero': np.std(non_zero_coeffs) if num_non_zero > 0 else 0
    }
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    LASSO HOMOTOPY SPARSITY REPORT
    {'='*60}
    Model Parameters:
    - lambda_par (λ): {model.lambda_par:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    
    Sparsity Summary:
    - Total features: {len(coefficients)}
    - Non-zero coefficients: {num_non_zero} ({num_non_zero/len(coefficients):.1%})
    - Zero coefficients: {num_zero} ({sparsity:.1%})
    
    Coefficient Statistics:
    - Largest absolute coefficient: {coeff_stats['max']:.4f}
    - Smallest non-zero coefficient: {coeff_stats['min_nonzero']:.4f}
    - Mean absolute non-zero coefficient: {coeff_stats['mean_nonzero']:.4f}
    - Std of non-zero coefficients: {coeff_stats['std_nonzero']:.4f}
    
    Top 5 Largest Coefficients:
    {generate_coeff_table(coefficients, 5)}
    
    Bottom 5 Smallest Non-zero Coefficients:
    {generate_coeff_table(coefficients, -5) if num_non_zero > 5 else "N/A (not enough non-zero coefficients)"}
    {'='*60}
    """
    
    print(report)
    
    # Visualization
    plot_coefficient_distribution(coefficients, model.model.tol)
    
    # Assertion with helpful message
    assert sparsity > 0.1, (
        f"Sparsity test failed (expected >10%, got {sparsity:.1%}).\n"
        f"Suggested actions:\n"
        f"1. Increase lambda_par (current: {model.lambda_par})\n"
        f"2. Check feature correlations\n"
        f"3. Verify data standardization\n"
        f"4. Review tolerance setting (current: {model.model.tol:.2e})"
    )

# Helper functions for enhanced reporting
def generate_coeff_table(coefficients, n):
    """Generate formatted table of top/bottom coefficients"""
    if n > 0:  # Top n
        indices = np.argsort(-np.abs(coefficients))[:n]
    else:  # Bottom n
        non_zero = coefficients[np.abs(coefficients) > 1e-10]
        if len(non_zero) == 0:
            return "No non-zero coefficients"
        indices = np.argsort(np.abs(non_zero))[:abs(n)]
    
    rows = []
    for idx in indices:
        rows.append(f"    - Feature {idx:4d}: {coefficients[idx]:+.6f} (abs: {np.abs(coefficients[idx]):.6f})")
    return '\n'.join(rows)

def plot_coefficient_distribution(coefficients, tol):
    """Visualize coefficient distribution"""
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("coefficient distribution",fontsize=14, fontweight='bold', y=1.02)
    
    # Plot coefficient values
    plt.subplot(1, 2, 1)
    plt.stem(np.arange(len(coefficients)), coefficients, markerfmt=' ')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(tol, color='red', linestyle='--', alpha=0.5, label=f'Tolerance ({tol:.1e})')
    plt.axhline(-tol, color='red', linestyle='--', alpha=0.5)
    plt.title('Coefficient Values')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.legend()
    
    # Plot sorted absolute values
    plt.subplot(1, 2, 2)
    sorted_abs = np.sort(np.abs(coefficients))[::-1]
    plt.plot(sorted_abs, 'o-')
    plt.axhline(tol, color='red', linestyle='--', label=f'Tolerance ({tol:.1e})')
    plt.title('Sorted Absolute Coefficient Values')
    plt.xlabel('Rank')
    plt.ylabel('Absolute Coefficient Value')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../images/coefficientdistribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def test_prediction_consistency():
    """Comprehensive test of prediction consistency across multiple runs"""
    # Initialize model with fixed random state for reproducibility
    model = LassoHomotopyModel(lambda_par=0.1, fit_intercept=True, max_iter=1000)
    X, y = load_data()
    
    # Standardize features for consistent results
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # First fit and predict
    results1 = model.fit(X, y)
    preds1 = results1.predict(X)
    
    # Second fit and predict
    results2 = model.fit(X, y)
    preds2 = results2.predict(X)
    
    # Calculate differences and metrics
    diffs = np.abs(preds1 - preds2)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    diff_indices = np.where(diffs > 1e-4)[0]  # More stringent threshold
    passed = len(diff_indices) == 0
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    PREDICTION CONSISTENCY TEST REPORT
    {'='*60}
    Test Configuration:
    - Samples: {X.shape[0]}
    - Features: {X.shape[1]}
    - Random state: {'Fixed' if hasattr(model, 'random_state') else 'Not fixed'}
    - Tolerance threshold: 1.0e-04
    
    Consistency Metrics:
    - Maximum difference: {max_diff:.6f}
    - Mean difference: {mean_diff:.6f}
    - Standard deviation of differences: {std_diff:.6f}
    - Samples exceeding threshold: {len(diff_indices)}/{len(preds1)} ({len(diff_indices)/len(preds1):.1%})
    
    Prediction Statistics:
    - First run mean prediction: {np.mean(preds1):.6f}
    - Second run mean prediction: {np.mean(preds2):.6f}
    - Mean absolute difference: {mean_diff:.6f}
    {'='*60}
    """
    print(report)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    plt.suptitle("Prediction Consistency",fontsize=14, fontweight='bold', y=1.02)
    # Plot 1: Prediction comparison scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(preds1, preds2, alpha=0.6, color='blue')
    plt.plot([min(preds1), max(preds1)], [min(preds1), max(preds1)], 'r--')
    plt.title('Prediction Run 1 vs Run 2', fontsize=12)
    plt.xlabel('First Run Predictions', fontsize=10)
    plt.ylabel('Second Run Predictions', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Differences distribution
    plt.subplot(1, 2, 2)
    plt.hist(diffs, bins=30, color='green', alpha=0.7)
    plt.axvline(1e-4, color='red', linestyle='--', label='Tolerance threshold')
    plt.title('Prediction Differences Distribution', fontsize=12)
    plt.xlabel('Absolute Difference', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("../images/predictionconfig.png",dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show detailed differences if test fails
    if not passed:
        print("\nDETAILED DIFFERENCE ANALYSIS:")
        print("-"*50)
        print(f"{'Index':<8} {'Run 1':<12} {'Run 2':<12} {'Difference':<12} {'Rel.Diff(%)':<12}")
        print("-"*50)
        for i in diff_indices[:10]:  # Show first 10 differing samples
            rel_diff = 100 * diffs[i] / (0.5 * (abs(preds1[i]) + abs(preds2[i])))
            print(f"{i:<8} {preds1[i]:<12.6f} {preds2[i]:<12.6f} {diffs[i]:<12.6f} {rel_diff:<12.2f}")
    
    # Assertions with helpful messages
    assert passed, (
        f"Found {len(diff_indices)} inconsistent predictions\n"
        f"Maximum difference: {max_diff:.6f} (threshold: 1.0e-04)\n"
        "Possible causes:\n"
        "1. Non-deterministic algorithm components\n"
        "2. Numerical instability\n"
        "3. High condition number in data\n"
        "Recommended actions:\n"
        "1. Set random_state if available\n"
        "2. Increase max_iter for convergence\n"
        "3. Check feature scaling"
    )

def test_feature_importance():
    """Visualize feature importance/coefficients"""
    model = LassoHomotopyModel()
    X, y = load_data()
    
    results = model.fit(X, y)
    coefficients = results.coef_ 
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(coefficients)), coefficients)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Coefficients (Importance)')
    plt.grid(True)
    plt.savefig("../images/featureimportance.png",dpi=300, bbox_inches='tight')
    plt.show()
    
def test_update_model():
    """Test online updating of LassoHomotopy model with visualization"""
    # Set up visualization with proper matplotlib style
    plt.rcParams.update({
        'figure.titlesize': 14,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })
    
    # Initialize model
    model = LassoHomotopyModel(lambda_par=0.5, fit_intercept=True)
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Initial fit
    model.fit(X, y)
    initial_coef = model.coef_.copy()
    
    # Set up visualization
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Lasso Homotopy Coefficient Trajectories',fontsize=14, fontweight='bold', y=1.02)
    
    # Track coefficient changes
    coef_history = [initial_coef]
    update_labels = ['Initial Fit']
    
    # Define test cases
    test_cases = [
        {'name': 'Normal sample', 'x': np.random.randn(10), 'y': 3.8},
        {'name': 'Outlier sample', 'x': np.random.randn(10)*3, 'y': 10.0},
        {'name': 'Zero feature sample', 'x': np.zeros(10), 'y': 1.0},
        {'name': 'Batch update', 'x': np.random.randn(5, 10), 'y': np.random.randn(5)},
        {'name': 'lambda_par change', 'x': np.random.randn(10), 'y': 2.5, 'lambda_par': 0.2}
    ]
    
    # Run test cases
    for case in test_cases:
        try:
            lambda_par = case.get('lambda_par')
            model.update_model(case['x'], case['y'], lambda_par_new=lambda_par)
            coef_history.append(model.coef_.copy())
            update_labels.append(f"{case['name']}\nλ={model.lambda_par:.2f}")
            
            print(f"\nAfter {case['name']}:")
            print(f"lambda_par: {model.lambda_par:.2f}")
            print(f"Active features: {len(model.model.active_set)}")
            print(f"Max coef change: {np.max(np.abs(coef_history[-1] - coef_history[-2])):.4f}")
            
        except Exception as e:
            print(f"Failed on {case['name']}: {str(e)}")
            continue
    
    
    # Visualization 1: Coefficient trajectories
    coef_history = np.array(coef_history).T
    for i, coef_traj in enumerate(coef_history):
        ax1.plot(coef_traj, label=f'Feature {i}', marker='o', markersize=5, alpha=0.8)
    
    ax1.set_title('Coefficient Values Across Updates')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_xticks(range(len(update_labels)))
    ax1.set_xticklabels(update_labels, rotation=45, ha='right')
    ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Visualization 2: Coefficient magnitudes
    coef_magnitudes = np.abs(coef_history)
    im = ax2.imshow(coef_magnitudes, aspect='auto', cmap='viridis')
    ax2.set_title('Coefficient Magnitudes (Absolute Values)')
    ax2.set_ylabel('Feature Index')
    ax2.set_xlabel('Update Step')
    ax2.set_xticks(range(len(update_labels)))
    ax2.set_xticklabels(update_labels, rotation=45, ha='right')
    ax2.set_yticks(range(10))
    plt.colorbar(im, ax=ax2, label='Magnitude')
    
    plt.tight_layout()
    plt.savefig("../images/onlineupdate.png",dpi=300,bbox_inches='tight')
    plt.show()
    
    # Final assertions
    assert len(model.coef_) == 10
    assert model.model.X_.shape[1] == 10

def test_with_builtin_models():
    print("=== Testing LassoHomotopyModel vs. Scikit-learn Lasso ===")
    
    # 1. Generate Synthetic Data (Controlled Experiment)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Models
    homotopy_model = LassoHomotopyModel(lambda_par=1.0, fit_intercept=True, max_iter=1000)
    results = homotopy_model.fit(X_train, y_train)
    homotopy_preds = results.predict(X_test)

    lasso_model = Lasso(alpha=1.0, max_iter=1000, fit_intercept=True,random_state=42)
    lasso_model.fit(X_train, y_train)
    lasso_preds = lasso_model.predict(X_test)

    # 3. Performance Metrics
    def evaluate_model(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        sparsity = np.mean(y_pred == 0) if hasattr(y_pred, '__len__') else 0
        return {
            "Model": name,
            "MSE": mse,
            "MAE": mae,
            "R²": r2,
            "Sparsity (%)": sparsity * 100
        }

    results = [
        evaluate_model("LassoHomotopy", y_test, homotopy_preds),
        evaluate_model("Scikit-Lasso", y_test, lasso_preds)
    ]

    print("\n=== Performance Comparison ===")
    for res in results:
        print(
            f"{res['Model']:>12} | "
            f"MSE: {res['MSE']:.4f} | "
            f"MAE: {res['MAE']:.4f} | "
            f"R²: {res['R²']:.4f} | "
            f"Sparsity: {res['Sparsity (%)']:.2f}%"
        )

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.1,
        'figure.facecolor': '#121212',
        'axes.facecolor': '#1e1e1e',
        'axes.edgecolor': '0.8',
        'text.color': 'white'
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("Model Comparison: LassoHomotopy vs. Scikit-Lasso", 
                fontsize=14, fontweight='bold', y=1.02, color='white')

    # Color palette optimized for dark background
    colors = {
        'homotopy': '#4dacd1',  # Light blue
        'lasso': '#f1a340',     # Orange
        'reference': '#e0e0e0', # Light gray
        'highlight': '#4d4d4d'  # Dark gray
    }

    # --- Plot 1: Actual vs. Predicted ---
    axes[0].scatter(y_test, homotopy_preds, alpha=0.8, 
                label='LassoHomotopy', color=colors['homotopy'],
                edgecolor=colors['reference'], linewidth=0.3)
    axes[0].scatter(y_test, lasso_preds, alpha=0.8, 
                label='Scikit-Lasso', color=colors['lasso'],
                edgecolor=colors['reference'], linewidth=0.3)
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                linestyle='--', color=colors['reference'], linewidth=1.2, 
                label='Perfect Fit')
    axes[0].set_title("Actual vs. Predicted Values", pad=12, color='white')
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend(framealpha=0.2, edgecolor='0.8')

    # Add correlation coefficient annotation
    homotopy_r = np.corrcoef(y_test, homotopy_preds)[0,1]
    lasso_r = np.corrcoef(y_test, lasso_preds)[0,1]
    axes[0].text(0.05, 0.9, f'$r_{{Homotopy}}$ = {homotopy_r:.3f}', 
                transform=axes[0].transAxes, color=colors['homotopy'])
    axes[0].text(0.05, 0.83, f'$r_{{Lasso}}$ = {lasso_r:.3f}', 
                transform=axes[0].transAxes, color=colors['lasso'])

    # --- Plot 2: Residual Analysis ---
    axes[1].scatter(homotopy_preds, y_test - homotopy_preds, alpha=0.8, 
                color=colors['homotopy'], label='LassoHomotopy',
                edgecolor=colors['reference'], linewidth=0.3)
    axes[1].scatter(lasso_preds, y_test - lasso_preds, alpha=0.8, 
                color=colors['lasso'], label='Scikit-Lasso',
                edgecolor=colors['reference'], linewidth=0.3)
    axes[1].axhline(0, color=colors['reference'], linestyle='--', linewidth=1)
    axes[1].set_title("Residual Analysis", pad=12, color='white')
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals (Actual - Predicted)")
    axes[1].legend(framealpha=0.2, edgecolor='0.8')

    # Add residual statistics
    axes[1].text(0.05, 0.9, f'Homotopy MSE: {mean_squared_error(y_test, homotopy_preds):.2f}', 
                transform=axes[1].transAxes, color=colors['homotopy'])
    axes[1].text(0.05, 0.83, f'Lasso MSE: {mean_squared_error(y_test, lasso_preds):.2f}', 
                transform=axes[1].transAxes, color=colors['lasso'])

    # --- Plot 3: Coefficient Comparison ---
    width = 0.35
    x = np.arange(X.shape[1])
    axes[2].bar(x - width/2, homotopy_model.coef_, width, 
            label='LassoHomotopy', color=colors['homotopy'],
            edgecolor=colors['reference'], linewidth=0.5)
    axes[2].bar(x + width/2, lasso_model.coef_, width, 
            label='Scikit-Lasso', color=colors['lasso'],
            edgecolor=colors['reference'], linewidth=0.5)
    axes[2].axhline(0, color=colors['reference'], linestyle='-', linewidth=0.7)
    axes[2].set_title("Feature Coefficients Comparison", pad=12, color='white')
    axes[2].set_xlabel("Feature Index")
    axes[2].set_ylabel("Coefficient Value")
    axes[2].legend(framealpha=0.2, edgecolor='0.8')

    # Highlight important features
    axes[2].axhspan(-0.1, 0.1, facecolor=colors['highlight'], alpha=0.3, zorder=0)
    axes[2].text(0.02, 0.95, 'Gray band: Near-zero coefficients', 
                transform=axes[2].transAxes, color=colors['reference'], fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("../images/benchmarkcompare_dark.png", dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.show()


if __name__ == "__main__":
    test_basic_prediction()
    test_prediction_visualization()
    test_single_feature()
    test_high_dimensional_data()
    test_sparse_solution()
    test_prediction_consistency()
    test_feature_importance()
    test_update_model()
    test_with_builtin_models()
    print("All tests passed!")