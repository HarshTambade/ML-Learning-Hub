from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Seasonal Decomposition

def decompose_additive(ts, period=12):
    """
    Perform additive seasonal decomposition
    """
    decomposition = seasonal_decompose(ts, model='additive', period=period)
    return decomposition

def decompose_multiplicative(ts, period=12):
    """
    Perform multiplicative seasonal decomposition
    """
    decomposition = seasonal_decompose(ts, model='multiplicative', period=period)
    return decomposition

def plot_decomposition(decomposition):
    """
    Visualize decomposition components
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create synthetic data
    t = np.arange(120)
    trend = t * 0.1
    seasonal = 10 * np.sin(2*np.pi*t/12)
    noise = np.random.randn(120) * 2
    ts = pd.Series(trend + seasonal + noise)
    
    # Decompose
    add_decomp = decompose_additive(ts, period=12)
    mult_decomp = decompose_multiplicative(ts, period=12)
    
    # Visualize
    plot_decomposition(add_decomp)
    print("Decomposition Complete")
    print(f"Trend: {add_decomp.trend.dropna().values[:5]}")
    print(f"Seasonal: {add_decomp.seasonal.values[:12]}")
    print(f"Residual mean: {add_decomp.resid.mean():.4f}")
