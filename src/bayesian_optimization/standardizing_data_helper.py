import numpy as np
from scipy.special import logit, expit


# Default values for BP and GC
DEFAULTS = {
    'BP': {'mu': 4.48, 'sig': 0.75},
    'GC': {'mu': -0.282, 'sig': 1},
}

def skip(x):
    """Identity transform"""
    return x

# Define transforms for each variable
TRANSFORMS = {
    'BP': [np.log, np.exp],
    'GC': [logit, expit],
    'r': [skip, skip]
}

def stdz(x, transform, mu, sigma):
    """Standardize: transform, mean-center, scale"""
    x_ = transform[0](x)
    return (x_ - mu) / sigma

def unstdz(z, transform, mu, sigma):
    """Unstandardize: rescale, un-center, invert transform"""
    x_ = z * sigma + mu
    return transform[1](x_)

def compute_r_stats(data):
    """Compute mean and std for 'Value' column"""
    return {'mu': data['Value'].mean(), 'sig': data['Value'].std()}

def standardize_dataframe(data, r_stdz=None):
    """
    Standardize and unstandardize BP, GC, and r.
    Adds new columns to dataframe.
    """
    if r_stdz is None:
        r_stdz = compute_r_stats(data)

    data['r_stdz'] = stdz(data['Value'], TRANSFORMS['r'], r_stdz['mu'], r_stdz['sig'])
    data['BP_stdz'] = stdz(data['BP'], TRANSFORMS['BP'], DEFAULTS['BP']['mu'], DEFAULTS['BP']['sig'])
    data['GC_stdz'] = stdz(data['GC'], TRANSFORMS['GC'], DEFAULTS['GC']['mu'], DEFAULTS['GC']['sig'])

    data['r_unstdz'] = unstdz(data['r_stdz'], TRANSFORMS['r'], r_stdz['mu'], r_stdz['sig'])
    data['BP_unstdz'] = unstdz(data['BP_stdz'], TRANSFORMS['BP'], DEFAULTS['BP']['mu'], DEFAULTS['BP']['sig'])
    data['GC_unstdz'] = unstdz(data['GC_stdz'], TRANSFORMS['GC'], DEFAULTS['GC']['mu'], DEFAULTS['GC']['sig'])

    # Optional consistency check
    assert np.allclose(data['Value'], data['r_unstdz'], atol=1e-10)
    assert np.allclose(data['BP'], data['BP_unstdz'], atol=1e-10)
    assert np.allclose(data['GC'], data['GC_unstdz'], atol=1e-10)

    return data, r_stdz
