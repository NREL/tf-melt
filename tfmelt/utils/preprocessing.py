from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class IdentityScaler:
    """
    A scaler that performs no scaling, behaving like an identity function.

    This class is useful for pipelines where a scaler is optional, but the pipeline
    expects a fit and transform method to be present like in Scikit-learn.
    """

    def __init__(self, **kwargs):
        self.scale_ = 1.0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        return X


def get_normalizers(norm_type="standard", n_normalizers=1, **kwargs):
    """
    Get a list of normalizers based on the specified normalization type.

    Parameters:
    norm_type (str): Type of normalization ('standard', 'minmax', 'robust', 'power', 'quantile').
    n_normalizers (int): Number of normalizers to create. Defaults to 1.
    **kwargs: Additional keyword arguments for the specific scaler.

    Returns:
    list: A list of normalizers.
    """
    normalizers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
        "none": IdentityScaler,
    }

    if norm_type not in normalizers:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    scaler_class = normalizers[norm_type]

    # Extract relevant kwargs for each scaler
    scaler_params = {
        "minmax": ["feature_range"],
        "quantile": ["output_distribution", "n_quantiles", "random_state"],
        "power": ["method", "standardize"],
    }

    relevant_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in scaler_params.get(norm_type, [])
    }

    # Create the specified number of normalizers
    normalizers_list = [scaler_class(**relevant_kwargs) for _ in range(n_normalizers)]

    return normalizers_list
