import tensorflow.keras.initializers as initializers


def get_initializer(init_name: str, **kwargs):
    """Utility method to get initializer based on its name and seed."""
    if init_name == "glorot_uniform":
        return initializers.GlorotUniform(**kwargs)
    elif init_name == "glorot_normal":
        return initializers.GlorotNormal(**kwargs)
    elif init_name == "he_normal":
        return initializers.HeNormal(**kwargs)
    elif init_name == "he_uniform":
        return initializers.HeUniform(**kwargs)
    elif init_name == "lecun_normal":
        return initializers.LecunNormal(**kwargs)
    elif init_name == "lecun_uniform":
        return initializers.LecunUniform(**kwargs)
    elif init_name == "random_normal":
        return initializers.RandomNormal(**kwargs)
    elif init_name == "random_uniform":
        return initializers.RandomUniform(**kwargs)
    elif init_name == "constant":
        return initializers.Constant(**kwargs)
    elif init_name == "identity":
        return initializers.Identity(**kwargs)
    elif init_name == "ones":
        return initializers.Ones(**kwargs)
    elif init_name == "orthogonal":
        return initializers.Orthogonal(**kwargs)
    elif init_name == "zeros":
        return initializers.Zeros(**kwargs)
    elif init_name == "variance_scaling":
        return initializers.VarianceScaling(**kwargs)
    elif init_name == "truncated_normal":
        return initializers.TruncatedNormal(**kwargs)
    else:
        raise ValueError(f"Unsupported initializer {init_name}")
