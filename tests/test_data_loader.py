import unittest
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data_shapes(self):
        X_train, X_test, y_train, y_test, feature_cols = load_data()
        self.assertFalse(X_train.empty)
        self.assertFalse(X_test.empty)
        self.assertEqual(len(X_train.columns), len(feature_cols))

if __name__ == "__main__":
    unittest.main()
