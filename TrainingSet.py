from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, y_train  # 只返回训练数据

