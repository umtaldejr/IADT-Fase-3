#!/usr/bin/env python3
"""
Script to generate standardized loss plots for all versions in metrics/ folder

This script creates:
1. Individual loss plots for each version with consistent styling
2. A comparison plot showing all versions together

Features:
- 5% window moving average for smoothing
- Discrete circle markers for minimum loss points
- Clean statistics box with min loss step information
- Professional styling with subtle colors
- Consistent layout across all versions

Usage:
    python generate_loss_plots.py

Requirements:
    - pandas
    - matplotlib
    - numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import sys

def create_individual_loss_plot(csv_file, version_name):
    """
    Create a standardized loss plot for a given CSV file
    
    Args:
        csv_file (str): Path to the CSV file containing training data
        version_name (str): Version name (e.g., 'v1.5.0')
    
    Returns:
        dict: Statistics about the training run
    """
    
    # Read the CSV data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None
    
    # Validate data structure
    required_columns = ['Step', 'Training Loss']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: {csv_file} missing required columns: {required_columns}")
        return None
    
    # Calculate 5% window size for moving average
    total_points = len(data)
    window_size = max(1, int(total_points * 0.05))  # 5% of total data points
    
    # Create the plot with consistent style
    plt.figure(figsize=(12, 6))
    
    # Plot the training loss with light blue color
    plt.plot(data['Step'], data['Training Loss'], color='#5B9BD5', linewidth=1.2, label='Training Loss')
    
    # Add moving average with 5% window
    moving_avg = data['Training Loss'].rolling(window=window_size, center=False).mean()
    plt.plot(data['Step'], moving_avg, color='#E74C3C', linewidth=2, label=f'Moving Average (window={window_size})')
    
    # Find minimum loss point
    min_loss_idx = data['Training Loss'].idxmin()
    min_loss_step = data.loc[min_loss_idx, 'Step']
    min_loss_value = data.loc[min_loss_idx, 'Training Loss']
    
    # Add minimum loss marker with discrete circle
    plt.scatter(min_loss_step, min_loss_value, color='#FF6B35', s=60, zorder=5, 
                marker='o', edgecolors='#333333', linewidth=1.5, alpha=0.8, 
                label=f'Min Loss: {min_loss_value:.4f}')
    
    # Calculate other statistics
    max_loss = data['Training Loss'].max()
    mean_loss = data['Training Loss'].mean()
    final_loss = data['Training Loss'].iloc[-1]
    
    # Add statistics box with step information included
    stats_text = f'Min: {min_loss_value:.4f} @{min_loss_step}\nMax: {max_loss:.4f}\nMean: {mean_loss:.4f}\nFinal: {final_loss:.4f}'
    
    # Position the text box in the upper left corner
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='black'),
             fontsize=10, fontfamily='monospace')
    
    # Customize the plot
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend in the upper right
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits
    plt.xlim(0, data['Step'].max())
    plt.ylim(0, max_loss * 1.05)  # Add 5% padding at the top
    
    # Remove top and right spines for cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the plot
    output_path = Path(f"metrics/{version_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'version': version_name,
        'total_points': total_points,
        'window_size': window_size,
        'min_loss': min_loss_value,
        'min_loss_step': min_loss_step,
        'max_loss': max_loss,
        'mean_loss': mean_loss,
        'final_loss': final_loss
    }

def create_comparison_plot():
    """
    Create a comparison plot of all versions with minimum loss markers
    
    Returns:
        list: Information about minimum losses for each version
    """
    
    # Find all CSV files in metrics folder
    csv_files = glob.glob("metrics/v*.csv")
    if not csv_files:
        print("No CSV files found in metrics/ folder")
        return []
    
    csv_files.sort()  # Sort to process in version order
    
    # Create the comparison plot
    plt.figure(figsize=(15, 8))
    
    # Color palette for different versions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    max_steps = 0
    min_markers = []
    
    for i, csv_file in enumerate(csv_files):
        # Extract version name from filename
        version_name = Path(csv_file).stem  # e.g., "v1.1.0" from "v1.1.0.csv"
        
        try:
            # Read the CSV data
            data = pd.read_csv(csv_file)
            
            # Calculate 5% window size for moving average
            total_points = len(data)
            window_size = max(1, int(total_points * 0.05))
            
            # Plot raw data with transparency
            plt.plot(data['Step'], data['Training Loss'], 
                    color=colors[i % len(colors)], alpha=0.3, linewidth=0.5)
            
            # Plot moving average
            moving_avg = data['Training Loss'].rolling(window=window_size, center=False).mean()
            plt.plot(data['Step'], moving_avg, 
                    color=colors[i % len(colors)], linewidth=2, 
                    label=f'{version_name} (avg)')
            
            # Find and mark minimum loss point with discrete circles
            min_loss_idx = data['Training Loss'].idxmin()
            min_loss_step = data.loc[min_loss_idx, 'Step']
            min_loss_value = data.loc[min_loss_idx, 'Training Loss']
            
            plt.scatter(min_loss_step, min_loss_value, color=colors[i % len(colors)], 
                       s=50, zorder=5, marker='o', edgecolors='#333333', linewidth=1.2, alpha=0.8)
            
            min_markers.append({
                'version': version_name,
                'step': min_loss_step,
                'loss': min_loss_value,
                'color': colors[i % len(colors)]
            })
            
            # Track maximum steps for axis limits
            max_steps = max(max_steps, data['Step'].max())
            
        except Exception as e:
            print(f"Error processing {csv_file} for comparison plot: {e}")
            continue
    
    if not min_markers:
        print("No valid data found for comparison plot")
        return []
    
    # Add text box with all minimum losses including step information
    min_text = "Minimum Losses:\n" + "\n".join([
        f"{m['version']}: {m['loss']:.4f} @{m['step']}" for m in min_markers
    ])
    
    plt.text(0.02, 0.98, min_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black'),
             fontsize=10, fontfamily='monospace', color='black')
    
    # Customize the plot
    plt.title('Training Loss Comparison - All Versions', fontsize=16, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits
    plt.xlim(0, max_steps)
    plt.ylim(0, 4.5)  # Set reasonable upper limit
    
    # Remove top and right spines for cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the plot
    output_path = Path("metrics/all_versions_loss_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return min_markers

def main():
    """
    Main function to generate all loss plots
    """
    
    print("Generating standardized loss plots...")
    print("=" * 60)
    
    # Check if metrics directory exists
    metrics_dir = Path("metrics")
    if not metrics_dir.exists():
        print("Error: metrics/ directory not found")
        sys.exit(1)
    
    # Find all CSV files in metrics folder
    csv_files = glob.glob("metrics/v*.csv")
    if not csv_files:
        print("No CSV files found in metrics/ folder")
        sys.exit(1)
    
    csv_files.sort()  # Sort to process in version order
    
    results = []
    
    # Generate individual plots
    print(f"Found {len(csv_files)} CSV files to process")
    print("-" * 60)
    
    for csv_file in csv_files:
        # Extract version name from filename
        version_name = Path(csv_file).stem  # e.g., "v1.1.0" from "v1.1.0.csv"
        
        print(f"Processing {version_name}...")
        
        result = create_individual_loss_plot(csv_file, version_name)
        if result:
            results.append(result)
            print(f"  ✓ Created {version_name}_loss_plot.png")
            print(f"    Data points: {result['total_points']}")
            print(f"    Window size: {result['window_size']} ({result['window_size']/result['total_points']*100:.1f}%)")
            print(f"    Min loss: {result['min_loss']:.4f} at step {result['min_loss_step']}")
        else:
            print(f"  ✗ Failed to process {version_name}")
        print()
    
    # Generate comparison plot
    print("Creating comparison plot...")
    min_markers = create_comparison_plot()
    if min_markers:
        print("  ✓ Created all_versions_loss_comparison.png")
    else:
        print("  ✗ Failed to create comparison plot")
    
    # Print summary
    print("=" * 60)
    print("Summary of all versions:")
    print("-" * 60)
    
    if results:
        for result in results:
            print(f"{result['version']:>8} | Points: {result['total_points']:>4} | "
                  f"Window: {result['window_size']:>3} | "
                  f"Final Loss: {result['final_loss']:>7.4f} | "
                  f"Min Loss: {result['min_loss']:>7.4f} @{result['min_loss_step']}")
        
        print(f"\nSuccessfully processed {len(results)} versions!")
    else:
        print("No versions were successfully processed.")
        sys.exit(1)
    
    print("\nAll plots generated with:")
    print("- Consistent styling and colors")
    print("- 5% window moving averages")
    print("- Discrete circle markers for minimum loss")
    print("- Clean statistics boxes")
    print("- Professional layout")

if __name__ == "__main__":
    main()
