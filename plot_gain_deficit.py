import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union

def generate_complex_normal(size: Tuple[int, ...], rho: float = 0.0) -> np.ndarray:
    """
    Generate complex Gaussian random variables with AR(1) correlation structure.
    Args:
        size (Tuple[int, ...]): Shape of the output array (N, ...).
        rho (float): Correlation coefficient (0 <= rho < 1).
    Returns:
        np.ndarray: Array of complex Gaussian random variables with AR(1) correlation.
    """
    iid = (np.random.randn(*size) + 1j * np.random.randn(*size)) / np.sqrt(2.0)
    if rho == 0.0:
        return iid
    N = size[0]
    h = np.zeros(size, dtype=np.complex128)
    h[0] = iid[0]
    scale = np.sqrt(1 - np.abs(rho)**2)
    for n in range(1, N):
        h[n] = rho * h[n-1] + scale * iid[n]
    return h

def get_antenna_weight_set(config: Union[int, str] = 4) -> np.ndarray:
    if isinstance(config, int):
        antenna_weight_set = np.exp(1j * 2 * np.pi * np.arange(config) / config)
    elif config == 'binary':
        antenna_weight_set = np.array([0.0 + 0j, 1.0 + 0j])
    elif config == 'example':
        amplitudes = np.array([0.9, 0.75, 0.9, 0.9, 0.8, 0.7, 0.8])
        phases = np.array([10, 60, 100, 140, 220, 260, 315])
        antenna_weight_set = amplitudes * np.exp(1j * np.deg2rad(phases))

    else:
        raise ValueError(f"Unknown phase configuration: {config}")
    return antenna_weight_set

def compute_perimeter(antenna_weight_set: np.ndarray) -> float:
    """Computes the perimeter of a phase set, which should be arranged in counter-clockwise order."""
    antenna_weight_set_augmented = np.concatenate([antenna_weight_set, [antenna_weight_set[0]]])
    return np.sum(np.abs(np.diff(antenna_weight_set_augmented)))

def compute_ideal_gain(h: np.ndarray) -> np.ndarray:
    """Computes the ideal gain."""
    return np.sum(np.abs(h), axis=0)

def compute_antennawise_gain(h: np.ndarray, antenna_weight_set: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the antennawise gain
    Args:
        h (np.ndarray): (num_antennas, num_trials)
        antenna_weight_set (np.ndarray): (num_weights,)
    Returns:
        Tuple[np.ndarray, np.ndarray]: (num_trials,)
    """
    idx = np.argmax(np.real(h[..., None] * antenna_weight_set), axis=-1)
    w = antenna_weight_set[idx]
    return np.abs(np.sum(w * h, axis=0)), idx

def compute_optimum_gain(h: np.ndarray, antenna_weight_set: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the optimum gain
    Args:
        h (np.ndarray): (num_antennas,)
        antenna_weight_set (np.ndarray): (num_weights,)
    Returns:
        Tuple[float, np.ndarray]: Gain and weight indices (num_antennas,)
    """
    num_antennas, num_weights = len(h), len(antenna_weight_set)
    if num_antennas == 0 or num_weights == 0: return 0.0, np.zeros(0, dtype=int)
    points = h[:, None] * antenna_weight_set[None, :]
    start_idx = np.argmin(points.imag, axis=1)
    order_idx = (np.arange(num_weights)[None, :] + start_idx[:, None]) % num_weights
    polygons = points[np.arange(num_antennas)[:, None], order_idx]
    edges = np.roll(polygons, -1, axis=1) - polygons
    edge_angles = np.mod(np.angle(edges), 2 * np.pi)
    sort_idx = np.argsort(edge_angles.ravel(), kind='stable')
    sorted_edges = edges.ravel()[sort_idx]
    sorted_poly_indices = np.repeat(np.arange(num_antennas), num_weights)[sort_idx]
    trajectory = np.sum(polygons[:, 0]) + np.cumsum(sorted_edges)
    magnitudes = np.abs(trajectory)
    max_idx = np.argmax(magnitudes)
    best_val = max(np.abs(np.sum(polygons[:, 0])), magnitudes[max_idx])
    best_step = max_idx if magnitudes[max_idx] > np.abs(np.sum(polygons[:, 0])) else -1
    shifts = np.zeros(num_antennas, dtype=int) if best_step == -1 else np.bincount(sorted_poly_indices[:best_step+1], minlength=num_antennas)
    return float(best_val), order_idx[np.arange(num_antennas), shifts % num_weights]

def simulate_metrics(N, trials=10000, antenna_weight_config='example', method='antennawise', seed=None, rho=0.0):
    if seed is not None: np.random.seed(seed)
    antenna_weight_set = get_antenna_weight_set(config=antenna_weight_config)
    
    # Process trials in batches to keep memory usage low
    batch_size = 10000
    ratios = []
    
    for start_trial in range(0, trials, batch_size):
        this_batch = min(batch_size, trials - start_trial)
        h_batch = generate_complex_normal((N, this_batch), rho=rho)
        ideal_gains = compute_ideal_gain(h_batch)
        
        if method == 'antennawise':
            discrete_gains, _ = compute_antennawise_gain(h_batch, antenna_weight_set)
        elif method == 'optimum':
            discrete_gains = np.array([compute_optimum_gain(h_batch[:, i], antenna_weight_set)[0] for i in range(this_batch)])
        else:
            raise ValueError(f"Unknown method: {method}")
            
        ratios.append(discrete_gains / ideal_gains)
        
    return np.concatenate(ratios), antenna_weight_set

if __name__ == "__main__": 
    import csv
    import os

    N_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    TRIALS, SEED = 100000, 1234
    PCT_LOW, PCT_HIGH = 10, 90

    CONFIGS = [
        ('binary', 'binary'),
        (4,        'square'),
        ('example', 'example')
    ]
    RHOS = [
        (0.0, 'uncorrelated'),
        (0.9, 'correlated')
    ]

    for weight_config, weight_name in CONFIGS:
        for rho_val, rho_name in RHOS:
            csv_name = f"{weight_name}_{rho_name}.csv"
            print(f"Simulating {csv_name}...", flush=True)
            
            data = {'opt': {'mean': [], 'p_low': [], 'p_high': []}, 
                    'ant': {'mean': [], 'p_low': [], 'p_high': []}}
            
            # Get the weight set once for calculating the bound
            antenna_weight_set = get_antenna_weight_set(weight_config)
            bound_val = 10 * np.log10((compute_perimeter(antenna_weight_set) / (2 * np.pi))**2)
            
            for N in N_values:
                print(f"  N={N}...", end=" ", flush=True)
                for method in ['opt', 'ant']:
                    r, _ = simulate_metrics(
                        N, trials=TRIALS, antenna_weight_config=weight_config, 
                        method='optimum' if method == 'opt' else 'antennawise', 
                        seed=SEED, rho=rho_val
                    )
                    # Average of squared ratios, then to dB
                    data[method]['mean'].append(10 * np.log10(np.mean(r**2)))
                    # Percentiles of squared ratios, then to dB
                    data[method]['p_low'].append(10 * np.log10(np.percentile(r**2, PCT_LOW)))
                    data[method]['p_high'].append(10 * np.log10(np.percentile(r**2, PCT_HIGH)))
                print("done.", flush=True)
            
            # Save to CSV
            with open(csv_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['N', 'opt_mean', 'opt_low', 'opt_high', 'ant_mean', 'ant_low', 'ant_high', 'bound'])
                for i, n in enumerate(N_values):
                    writer.writerow([
                        n,
                        data['opt']['mean'][i], data['opt']['p_low'][i], data['opt']['p_high'][i],
                        data['ant']['mean'][i], data['ant']['p_low'][i], data['ant']['p_high'][i],
                        bound_val
                    ])
            print(f"  Saved to {csv_name}")

    print("All simulations complete.")