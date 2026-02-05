"""
Data loading and management utilities for capstone project
"""

import numpy as np


def load_initial_data(base_path="../initial_data"):
    """
    Load initial data for all 8 functions

    Args:
        base_path: Path to initial_data directory

    Returns:
        inputs: Dict of input arrays {func_id: array}
        outputs: Dict of output arrays {func_id: array}
    """
    inputs = {}
    outputs = {}

    for i in range(1, 9):
        folder = f"{base_path}/function_{i}"
        inputs[i] = np.load(f"{folder}/initial_inputs.npy")
        outputs[i] = np.load(f"{folder}/initial_outputs.npy")

    return inputs, outputs


def combine_with_week_results(inputs, outputs, week_inputs, week_outputs):
    """
    Combine existing data with results from a week

    Args:
        inputs: Dict of current input arrays {func_id: array}
        outputs: Dict of current output arrays {func_id: array}
        week_inputs: Dict of week's input points {func_id: array}
        week_outputs: Dict of week's output values {func_id: float}

    Returns:
        combined_inputs: Dict of combined input arrays
        combined_outputs: Dict of combined output arrays
    """
    combined_inputs = {}
    combined_outputs = {}

    for i in range(1, 9):
        # Reshape week input to 2D if needed
        week_input_reshaped = week_inputs[i].reshape(1, -1)

        # Combine
        combined_inputs[i] = np.vstack([inputs[i], week_input_reshaped])
        combined_outputs[i] = np.append(outputs[i], week_outputs[i])

    return combined_inputs, combined_outputs


def print_data_summary(inputs, outputs, title="Data Summary"):
    """
    Print summary of data for all functions

    Args:
        inputs: Dict of input arrays
        outputs: Dict of output arrays
        title: Title for the summary
    """
    print(f"\n{title}:")
    print("=" * 60)
    for i in range(1, 9):
        n_points = inputs[i].shape[0]
        n_dims = inputs[i].shape[1]
        best_val = np.max(outputs[i])
        print(f"  Function {i}: {n_points} points, {n_dims}D, best = {best_val:.6f}")
    print("=" * 60)


def save_week_data(inputs, outputs, filename):
    """
    Save combined data for a week

    Args:
        inputs: Dict of input arrays
        outputs: Dict of output arrays
        filename: Output filename (e.g., "week2_data.npz")
    """
    # Convert dicts to format suitable for npz
    save_dict = {}
    for i in range(1, 9):
        save_dict[f'inputs_{i}'] = inputs[i]
        save_dict[f'outputs_{i}'] = outputs[i]

    np.savez(filename, **save_dict)
    print(f"\nData saved to {filename}")


def load_week_data(filename):
    """
    Load combined data from a week

    Args:
        filename: Input filename (e.g., "week2_data.npz")

    Returns:
        inputs: Dict of input arrays
        outputs: Dict of output arrays
    """
    data = np.load(filename)

    inputs = {}
    outputs = {}

    for i in range(1, 9):
        inputs[i] = data[f'inputs_{i}']
        outputs[i] = data[f'outputs_{i}']

    print(f"\nData loaded from {filename}")
    return inputs, outputs
