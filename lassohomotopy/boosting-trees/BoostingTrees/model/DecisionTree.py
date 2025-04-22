import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        """
        Node class for decision tree.
        
        Parameters:
        -----------
        feature_idx : int or None
            The index of the feature used for splitting
        threshold : float or None
            The threshold value for the split
        left : Node or None
            The left child node
        right : Node or None
            The right child node
        value : float or None
            The prediction value for leaf nodes
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    """
    Decision Tree Regressor for gradient boosting.
    Implements a simple regression tree that works with the GradientBoostingClassifier.
    """
    
    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Initialize the decision tree regressor.
        
        Parameters:
        -----------
        max_depth : int, default=3
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        """
        Build a decision tree regressor from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns:
        --------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Build the tree recursively
        self.root = self._grow_tree(X, y, depth=0)
        return self
        
    def _grow_tree(self, X, y, depth):
        """
        Build the tree by recursively finding the best split.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        depth : int
            Current depth of the tree.
            
        Returns:
        --------
        node : Node
            The root node of the subtree.
        """
        n_samples, n_features = X.shape
        
        # Check if we should stop splitting
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.all(np.abs(y - np.mean(y)) < 1e-6)):
            
            # Create a leaf node
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        
        # If no good split was found, create a leaf node
        if feature_idx is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        # Check if split is valid (both sides have samples)
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Return the decision node
        return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
    
    def _best_split(self, X, y):
        """
        Find the best split that minimizes the MSE.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns:
        --------
        best_feature : int or None
            The index of the best feature for splitting.
        best_threshold : float or None
            The threshold for the best split.
        """
        n_samples, n_features = X.shape
        
        # If not enough samples to split, return None
        if n_samples < self.min_samples_split:
            return None, None
        
        # Calculate the initial MSE (variance)
        parent_mse = np.var(y) * len(y)
        
        # Initialize variables to track the best split
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        
        # Loop through each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Skip if only one unique value
            if len(thresholds) <= 1:
                continue
            
            # Calculate potential split points (midpoints between consecutive values)
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Skip if one side is empty
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate the left and right MSE
                left_mse = np.var(y[left_indices]) * np.sum(left_indices) if np.sum(left_indices) > 1 else 0
                right_mse = np.var(y[right_indices]) * np.sum(right_indices) if np.sum(right_indices) > 1 else 0
                
                # Calculate the gain (reduction in MSE)
                gain = parent_mse - (left_mse + right_mse)
                
                # Update the best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def predict(self, X):
        """
        Predict regression target for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        X = np.asarray(X)
        return np.array([self._predict_sample(sample) for sample in X])
    
    def _predict_sample(self, sample):
        """
        Predict the target value for a single sample.
        
        Parameters:
        -----------
        sample : array-like of shape (n_features,)
            A single sample.
            
        Returns:
        --------
        value : float
            The predicted value.
        """
        node = self.root
        
        # Traverse the tree until reaching a leaf node
        while node.left is not None:
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value