import numpy as np
from .DecisionTree import DecisionTreeRegressor

class GradientBoostingClassifier:
    """
    Gradient Boosting Tree Classifier implemented from first principles.
    Based on the algorithm described in "The Elements of Statistical Learning" 
    by Hastie, Tibshirani, and Friedman (Sections 10.9-10.10)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize the gradient boosting classifier.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of boosting stages (trees) to use.
        learning_rate : float, default=0.1
            The learning rate shrinks the contribution of each tree.
        max_depth : int, default=3
            Maximum depth of each regression tree.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.F0 = None  # Initial prediction
        
    def _sigmoid(self, x):
        """Apply sigmoid function to input."""
        return 1 / (1 + np.exp(-x))
    
    def _log_loss_gradient(self, y, p):
        """
        Compute the negative gradient of log loss function.
        This is (y - p) for binary classification with log loss.
        
        Parameters:
        -----------
        y : array-like of shape (n_samples,)
            Target values (0 or 1).
        p : array-like of shape (n_samples,)
            Predicted probabilities.
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Negative gradient of log loss with respect to F.
        """
        return y - p
    
    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (0 or 1).
            
        Returns:
        --------
        self : object
        """
        # Ensure y is numpy array
        y = np.asarray(y)
        X = np.asarray(X)
        
        # Initialize F0 with log odds of the target mean
        # This is derived from the log loss optimization
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        
        # Handle edge cases to prevent division by zero
        if pos_count == 0:
            self.F0 = -10  # A large negative number to represent log(0)
        elif neg_count == 0:
            self.F0 = 10   # A large positive number to represent log(inf)
        else:
            self.F0 = np.log(pos_count / neg_count)
        
        # Initialize predictions with F0
        F = np.full(shape=len(y), fill_value=self.F0)
        
        # Boosting iterations
        for m in range(self.n_estimators):
            # Calculate current probabilities
            p = self._sigmoid(F)
            
            # Compute negative gradient (residuals)
            residuals = self._log_loss_gradient(y, p)
            
            # Fit a regression tree to the negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                        min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            
            # Update F with the prediction of the new tree
            update = self.learning_rate * tree.predict(X)
            F += update
            
            # Store the tree
            self.trees.append(tree)
            
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        proba : array-like of shape (n_samples, 2)
            Class probabilities of the input samples.
        """
        X = np.asarray(X)
        # Start with F0
        F = np.full(shape=len(X), fill_value=self.F0)
        
        # Add contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        proba = self._sigmoid(F)
        
        # Return probabilities for both classes
        return np.vstack([1 - proba, proba]).T
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)