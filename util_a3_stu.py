"""
util_a3_stu.py: Customerized Scalars 
"""
import numpy as np

def standard_scaler(
        X_train: np.ndarray,
        X_test: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return features standardized to zero mean and unit variance.

    Precondition
    ------------
        X_train is 2D with at least one row, and if X_test is
        given, X_test.shape[1] == X_train.shape[1]

    Parameters
    ----------
    X_train 
        Shape (n_samples, n_features)
        Training feature matrix. Mean and std are computed from this.
    X_test 
        Shape (m_samples, n_features)
        Test feature matrix. Transformed using training statistics.
    
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        X_train scaled, shape (n_samples, n_features), if X_test is None.
        Otherwise (X_train_scaled, X_test_scaled), where X_test_scaled
        has shape (m_samples, n_features).
    
    Returns
    -------
    X_train_scaled or (X_train_scaled, X_test_scaled)
        X_train_scaled 
            Shape (n_samples, n_features)
            Standardized training features with zero mean and unit variance.
        X_test_scaled
            Shape (m_samples, n_features), optional
            Standardized test features using training mean and std.

    Notes
    -----
    Fitting on training data only prevents data leakage — the test set
    statistics must never influence the transformation applied to either set.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_test is not None:
        X_test = np.asarray(X_test, dtype=np.float64)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std

    if X_test is None:
        return X_train_scaled

    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

    

def min_max_scaler(
        X_train: np.ndarray, 
        X_test: np.ndarray | None=None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return features scaled to [0, 1] range using 
       the minimum and maximum values computed from the training set only.

    Precondition
    ------------
        X_train is 2D with at least one row, and if X_test is
        given, X_test.shape[1] == X_train.shape[1]    

    Parameters
    ----------
    X_train 
        Shape (n_samples, n_features)
        Training feature matrix. Min and max are computed from this.
    X_test  
        Shape (m_samples, n_features)
        Test feature matrix. Transformed using training statistics.

    Returns
    -------
    X_train_scaled or (X_train_scaled, X_test_scaled)
        X_train_scaled
            Shape (n_samples, n_features)
            Scaled training features with values in [0, 1].
    X_test_scaled
        Shape (m_samples, n_features), optional
        Scaled test features using training min and max. Values may fall
        outside [0, 1] if test data contains values outside the training range.

    Notes
    -----
    Min and max are computed from training data only to prevent data leakage.
    Unlike StandardScaler, MinMaxScaler is sensitive to outliers since a single
    extreme value shifts the min or max and compresses all other values.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_test is not None:
        X_test = np.asarray(X_test, dtype=np.float64)

    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)

    max_distance = max_val - min_val
    max_distance[max_distance == 0] = 1

    X_train_scaled = (X_train - min_val) / max_distance

    if X_test is None:
        return X_train_scaled

    X_test_scaled = (X_test - min_val) / max_distance
    return X_train_scaled, X_test_scaled
    