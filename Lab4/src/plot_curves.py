#!/usr/bin/env python3
"""
Script to plot learning loss epoch curves and teacher forcing epoch curves
from training history data saved by the Trainer.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_training_history(history_path):
    """Load training history from JSON file"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def plot_learning_loss_curve(history, save_path=None):
    """Plot learning loss curve showing train and validation loss over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Loss over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning loss curve saved to: {save_path}")
    
    plt.show()
    return fig

def plot_teacher_forcing_curve(history, save_path=None):
    """Plot teacher forcing ratio curve over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history['epoch']
    tfr = history['teacher_forcing_ratio']
    
    ax.plot(epochs, tfr, 'g-', label='Teacher Forcing Ratio', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Teacher Forcing Ratio', fontsize=12)
    ax.set_title('Teacher Forcing Ratio over Epochs', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Teacher forcing curve saved to: {save_path}")
    
    plt.show()
    return fig

def plot_combined_curves(history, save_path=None):
    """Plot combined view of loss and teacher forcing ratio"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = history['epoch']
    
    # Loss subplot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Learning Loss and Teacher Forcing Ratio over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Teacher forcing subplot
    ax2.plot(epochs, history['teacher_forcing_ratio'], 'g-', label='Teacher Forcing Ratio', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Teacher Forcing Ratio', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined curves saved to: {save_path}")
    
    plt.show()
    return fig

def plot_kl_beta_curve(history, save_path=None):
    """Plot KL annealing beta curve over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history['epoch']
    beta = history['kl_beta']
    
    ax.plot(epochs, beta, 'm-', label='KL Annealing Beta', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Beta Value', fontsize=12)
    ax.set_title('KL Annealing Beta over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KL beta curve saved to: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from history data')
    parser.add_argument('--history_path', type=str, required=True, 
                       help='Path to training_history.json file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plot images (optional)')
    parser.add_argument('--plot_type', type=str, choices=['loss', 'tfr', 'beta', 'combined', 'all'], 
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"Error: History file not found at {args.history_path}")
        return
    
    # Load training history
    history = load_training_history(args.history_path)
    
    # Create save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate plots based on type
    if args.plot_type in ['loss', 'all']:
        save_path = os.path.join(args.save_dir, 'learning_loss_curve.png') if args.save_dir else None
        plot_learning_loss_curve(history, save_path)
    
    if args.plot_type in ['tfr', 'all']:
        save_path = os.path.join(args.save_dir, 'teacher_forcing_curve.png') if args.save_dir else None
        plot_teacher_forcing_curve(history, save_path)
    
    if args.plot_type in ['beta', 'all']:
        save_path = os.path.join(args.save_dir, 'kl_beta_curve.png') if args.save_dir else None
        plot_kl_beta_curve(history, save_path)
    
    if args.plot_type in ['combined', 'all']:
        save_path = os.path.join(args.save_dir, 'combined_curves.png') if args.save_dir else None
        plot_combined_curves(history, save_path)

if __name__ == '__main__':
    main()