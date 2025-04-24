import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import your gradient boosting implementation
from BoostingTrees.model.BoostingTree import GradientBoostingClassifier

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model, returning performance metrics."""
    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Time the prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # For your custom GradientBoostingClassifier
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        # If we have probabilities for multiple classes, take the positive class probability
        if y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
    else:
        y_prob = y_pred  # Fallback if predict_proba is not available
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'conf_matrix': conf_matrix
    }

def plot_comparison(results):
    """Plot a comparison of model performance metrics."""
    # Extract model names and metrics
    model_names = [result['model_name'] for result in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot performance metrics
    for i, metric in enumerate(metrics):
        values = [result[metric] for result in results]
        axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.4f}', ha='center')
    
    # Plot training time comparison
    train_times = [result['training_time'] for result in results]
    axes[5].bar(model_names, train_times)
    axes[5].set_title('Training Time (seconds)')
    for j, v in enumerate(train_times):
        axes[5].text(j, v + 0.01, f'{v:.4f}', ha='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('BoostingTrees/images/model_comparison.png')
    plt.show()

def main():
    """Main function to run the comparison."""
    print("Loading data...")
    try:
        # Try to load the training data
        data = pd.read_csv('BoostingTrees/tests/train_data.csv')
    except FileNotFoundError:
        print("Training data not found. Generating new data...")
        # Import the data generation function
        from generate_classification_data import generate_classification_data
        X, y = generate_classification_data(n_samples=1000, n_features=10)
        data = pd.DataFrame(np.column_stack([X, y]), 
                           columns=[f'feature_{i}' for i in range(X.shape[1])] + ['target'])
        data.to_csv('BoostingTrees/tests/train_data.csv', index=False)

    # Prepare the data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize models
    models = [
        {
            'model': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3),
            'name': 'Custom Gradient Boosting'
        },
        {
            'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4),
            'name': 'Custom GB (more trees, deeper)'
        },
        {
            'model': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            'name': 'Scikit-learn Random Forest'
        },
        {
            'model': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
            'name': 'RF (more trees, deeper)'
        }
    ]
    
    # Evaluate all models
    results = []
    for model_info in models:
        result = evaluate_model(
            model_info['model'], X_train, X_test, y_train, y_test, model_info['name']
        )
        results.append(result)
    
    # Plot the comparison
    plot_comparison(results)
    
    print("\nDetailed comparison:")
    # Show which model performed best for each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        best_idx = np.argmax([r[metric] for r in results])
        print(f"Best {metric}: {results[best_idx]['model_name']} ({results[best_idx][metric]:.4f})")

if __name__ == "__main__":
    main()