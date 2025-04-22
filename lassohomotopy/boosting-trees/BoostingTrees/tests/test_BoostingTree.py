import unittest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to sys.path to import model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.BoostingTree import GradientBoostingClassifier
from model.DecisionTree import DecisionTreeRegressor

class TestGradientBoostingClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Load small test dataset
        small_data_path = os.path.join(os.path.dirname(__file__), 'small_test.csv')
        if not os.path.exists(small_data_path):
            raise FileNotFoundError(f"Test file not found: {small_data_path}")
            
        data = pd.read_csv(small_data_path)
        cls.X = data.iloc[:, :-1].values
        cls.y = data.iloc[:, -1].values
        
        # Train a model with few estimators for quick tests
        cls.model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=2)
        cls.model.fit(cls.X, cls.y)
        
    def test_init(self):
        """Test model initialization."""
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.max_depth, 3)
        
    def test_sigmoid(self):
        """Test sigmoid function."""
        model = GradientBoostingClassifier()
        self.assertAlmostEqual(model._sigmoid(0), 0.5)
        self.assertTrue(0 < model._sigmoid(-10) < 0.01)
        self.assertTrue(0.99 < model._sigmoid(10) < 1)
        
    def test_gradient(self):
        """Test gradient calculation."""
        model = GradientBoostingClassifier()
        y = np.array([0, 1, 0, 1])
        p = np.array([0.3, 0.7, 0.6, 0.4])
        expected = np.array([-0.3, 0.3, -0.6, 0.6])
        np.testing.assert_almost_equal(model._log_loss_gradient(y, p), expected)
        
    def test_fit_predict(self):
        """Test that fit and predict work without errors."""
        # The model is already fitted in setUpClass
        predictions = self.model.predict(self.X)
        
        # Check that predictions have the right shape and values
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
        
    def test_predict_proba(self):
        """Test probability predictions."""
        probas = self.model.predict_proba(self.X)
        
        # Check shape and probability properties
        self.assertEqual(probas.shape, (len(self.X), 2))
        self.assertTrue(np.all(0 <= probas) and np.all(probas <= 1))
        np.testing.assert_almost_equal(np.sum(probas, axis=1), np.ones(len(self.X)))
        
    def test_decision_tree(self):
        """Test the decision tree component."""
        tree = DecisionTreeRegressor(max_depth=2)
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        tree.fit(X, y)
        
        # Test predictions
        predictions = tree.predict(X)
        self.assertEqual(len(predictions), len(y))
        
    def test_accuracy(self):
        """Test model accuracy on the training set."""
        # Predict on the training data
        y_pred = self.model.predict(self.X)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == self.y)
        print(f"Training accuracy: {accuracy:.4f}")
        
        # We expect better than random guessing
        self.assertTrue(accuracy > 0.5)
        
    def test_full_dataset(self):
        """Test on full dataset."""
        # Load the test dataset
        test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        if not os.path.exists(test_data_path):
            self.skipTest("Full test dataset not found")
            
        data = pd.read_csv(test_data_path)
        X_test = data.iloc[:, :-1].values
        y_test = data.iloc[:, -1].values
        
        # Create a fresh model with more estimators
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
        
        # Split the data for training
        n_train = len(X_test) // 2
        model.fit(X_test[:n_train], y_test[:n_train])
        
        # Predict on the remaining data
        y_pred = model.predict(X_test[n_train:])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test[n_train:])
        print(f"Test accuracy: {accuracy:.4f}")
        
        # We expect better than random guessing
        self.assertTrue(accuracy > 0.5)

if __name__ == '__main__':
    unittest.main()