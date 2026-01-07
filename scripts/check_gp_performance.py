
import jax
import jax.numpy as jnp
import sys
import time
from configs.default_config import config
from src.io.data_loader import load_all_data

def benchmark_cholesky():
    print("Loading data...")
    data_dict = load_all_data(config)
    
    # Simulate the construction of the full dataset used in posterior_predict
    input_xy_exp = data_dict["input_xy_exp"]
    input_xy_sim = data_dict["input_xy_sim"]
    
    # Concatenate experimental data
    exp_xy = jnp.concatenate(input_xy_exp, axis=0)
    
    print(f"Experimental Data Shape: {exp_xy.shape}")
    print(f"Simulation Data Shape: {input_xy_sim.shape}")
    
    # Combined training data for GP
    X = jnp.concatenate([exp_xy, input_xy_sim], axis=0)
    N = X.shape[0]
    print(f"Total GP Training Points (N): {N}")
    
    # Benchmark Cholesky
    print(f"Benchmarking Cholesky decomposition for {N}x{N} matrix...")
    
    # Create a dummy PD matrix
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (N, N))
    K = jnp.dot(A, A.T) + 1e-6 * jnp.eye(N)
    
    start_time = time.time()
    # Force compilation
    L = jnp.linalg.cholesky(K).block_until_ready()
    end_time = time.time()
    
    print(f"First run (compilation included): {end_time - start_time:.4f} s")
    
    start_time = time.time()
    for _ in range(10):
        L = jnp.linalg.cholesky(K).block_until_ready()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Average time (10 runs): {avg_time:.4f} s")
    
    prediction_samples = config["data"]["prediction_samples"]
    total_predictions = prediction_samples * 6 # 3 angles * 2 directions
    
    estimated_total_time = total_predictions * avg_time
    print(f"\nEstimated time for {total_predictions} predictions: {estimated_total_time:.1f} s ({estimated_total_time/60:.1f} min)")

if __name__ == "__main__":
    benchmark_cholesky()
