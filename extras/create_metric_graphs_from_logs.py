import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def parse_training_logs(log_file_path):
    """Parse training logs and extract all metrics"""
    
    data = {
        'train_batch_loss': [],
        'train_batch_steps': [],
        'train_epoch_loss': [],
        'train_epoch_steps': [],
        'val_batch_loss': [],
        'val_batch_steps': [],
        'val_epoch_loss': [],
        'val_epoch_steps': [],
        'learning_rate': [],
        'lr_steps': [],
        'grad_norm': [],
        'grad_norm_steps': [],
        'train_metrics': defaultdict(list),
        'train_metric_steps': [],
        'val_metrics': defaultdict(list),
        'val_metric_steps': []
    }
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip error lines
        if line.startswith('error'):
            continue
            
        # Parse training batch loss
        train_batch_match = re.search(r'\[(\d+)\]\[(\d+)/\d+\]\[(\d+)/\d+\] train loss: ([-\d.]+) \| lr: ([\d.]+) \| grad norm: ([\d.]+)', line)
        if train_batch_match:
            epoch, batch, global_step, loss, lr, grad_norm = train_batch_match.groups()
            data['train_batch_loss'].append(float(loss))
            data['train_batch_steps'].append(len(data['train_batch_steps']))  # Use sequential index
            data['learning_rate'].append(float(lr))
            data['lr_steps'].append(len(data['lr_steps']))  # Use sequential index
            data['grad_norm'].append(float(grad_norm))
            data['grad_norm_steps'].append(len(data['grad_norm_steps']))  # Use sequential index
            continue
            
        # Parse training epoch loss
        train_epoch_match = re.search(r'\[(\d+)\]\[(\d+)/\d+\] train epoch loss: ([-\d.]+)', line)
        if train_epoch_match:
            epoch, step, loss = train_epoch_match.groups()
            data['train_epoch_loss'].append(float(loss))
            data['train_epoch_steps'].append(len(data['train_epoch_steps']))  # Use sequential index
            continue
            
        # Parse validation batch loss
        val_batch_match = re.search(r'\[(\d+)\]\[(\d+)/\d+\]\[(\d+)/\d+\] valid loss: ([-\d.]+)', line)
        if val_batch_match:
            epoch, global_step, val_step, loss = val_batch_match.groups()
            data['val_batch_loss'].append(float(loss))
            data['val_batch_steps'].append(len(data['val_batch_steps']))  # Use sequential index
            continue
            
        # Parse validation epoch loss
        val_epoch_match = re.search(r'\[(\d+)\]\[(\d+)/\d+\] val epoch loss: ([-\d.]+)', line)
        if val_epoch_match:
            epoch, step, loss = val_epoch_match.groups()
            data['val_epoch_loss'].append(float(loss))
            data['val_epoch_steps'].append(len(data['val_epoch_steps']))  # Use sequential index
            continue
            
        # Parse training metrics
        train_metrics_match = re.search(r'\[(\d+)\] train \| (.+)', line)
        if train_metrics_match:
            epoch, metrics_str = train_metrics_match.groups()
            step = len(data['train_metric_steps'])  # Use sequential index
            data['train_metric_steps'].append(step)
            
            # Parse individual metrics
            metrics = re.findall(r'(\w+): ([\d.]+)', metrics_str)
            for metric_name, value in metrics:
                data['train_metrics'][metric_name].append(float(value))
            continue
            
        # Parse validation metrics
        val_metrics_match = re.search(r'\[(\d+)\] val \| (.+)', line)
        if val_metrics_match:
            epoch, metrics_str = val_metrics_match.groups()
            step = len(data['val_metric_steps'])  # Use sequential index
            data['val_metric_steps'].append(step)
            
            # Parse individual metrics
            metrics = re.findall(r'(\w+): ([\d.]+)', metrics_str)
            for metric_name, value in metrics:
                data['val_metrics'][metric_name].append(float(value))
            continue
    
    return data

def create_visualizations(data, viz_dir):
    """Create all visualizations and save them"""
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Training Batch Loss
    if data['train_batch_loss']:
        plt.figure(figsize=(14, 8))
        plt.plot(data['train_batch_steps'], data['train_batch_loss'], 
                linewidth=1, alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss (Batch-wise)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'train_batch_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Validation Batch Loss
    if data['val_batch_loss']:
        plt.figure(figsize=(14, 8))
        plt.plot(data['val_batch_steps'], data['val_batch_loss'], 
                linewidth=1, alpha=0.7, color='orange')
        plt.xlabel('Validation Step')
        plt.ylabel('Loss')
        plt.title('Validation Loss (Batch-wise)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'val_batch_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Training Epoch Loss
    if data['train_epoch_loss']:
        plt.figure(figsize=(12, 8))
        plt.plot(data['train_epoch_steps'], data['train_epoch_loss'], 
                'o-', linewidth=2, markersize=8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Epoch-wise)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'train_epoch_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Validation Epoch Loss
    if data['val_epoch_loss']:
        plt.figure(figsize=(12, 8))
        plt.plot(data['val_epoch_steps'], data['val_epoch_loss'], 
                's-', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss (Epoch-wise)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'val_epoch_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Learning Rate
    if data['learning_rate']:
        plt.figure(figsize=(12, 6))
        plt.plot(data['lr_steps'], data['learning_rate'], linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Gradient Norm
    if data['grad_norm']:
        plt.figure(figsize=(12, 6))
        plt.plot(data['grad_norm_steps'], data['grad_norm'], linewidth=1, alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Over Training')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'gradient_norm.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Training 3D Errors
    if data['train_metrics'] and any('3d_error' in k for k in data['train_metrics'].keys()):
        plt.figure(figsize=(12, 8))
        for metric_name in data['train_metrics']:
            if '3d_error' in metric_name:
                plt.plot(data['train_metric_steps'], data['train_metrics'][metric_name], 
                        'o-', label=metric_name, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('3D Error')
        plt.title('Training 3D Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'train_3d_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Validation 3D Errors
    if data['val_metrics'] and any('3d_error' in k for k in data['val_metrics'].keys()):
        plt.figure(figsize=(12, 8))
        for metric_name in data['val_metrics']:
            if '3d_error' in metric_name:
                plt.plot(data['val_metric_steps'], data['val_metrics'][metric_name], 
                        's-', label=metric_name, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('3D Error')
        plt.title('Validation 3D Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'val_3d_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. Training Motion Errors
    if data['train_metrics'] and any('motion_error' in k for k in data['train_metrics'].keys()):
        plt.figure(figsize=(12, 8))
        for metric_name in data['train_metrics']:
            if 'motion_error' in metric_name:
                plt.plot(data['train_metric_steps'], data['train_metrics'][metric_name], 
                        'o-', label=metric_name, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Motion Error')
        plt.title('Training Motion Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'train_motion_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 10. Validation Motion Errors
    if data['val_metrics'] and any('motion_error' in k for k in data['val_metrics'].keys()):
        plt.figure(figsize=(12, 8))
        for metric_name in data['val_metrics']:
            if 'motion_error' in metric_name:
                plt.plot(data['val_metric_steps'], data['val_metrics'][metric_name], 
                        's-', label=metric_name, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Motion Error')
        plt.title('Validation Motion Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'val_motion_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("All visualizations created successfully!")

def main():
    log_file_path = "/scratch/km6748/vision-experiments/job_output_61407798.txt"
    viz_dir = "/scratch/km6748/vision-experiments/extras/viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Reading log file: {log_file_path}")
    
    # Parse the logs
    try:
        data = parse_training_logs(log_file_path)
        print(f"Parsed {len(data['train_batch_loss'])} training batch losses")
        print(f"Parsed {len(data['val_batch_loss'])} validation batch losses")
        print(f"Parsed {len(data['train_epoch_loss'])} training epoch losses")
        print(f"Parsed {len(data['val_epoch_loss'])} validation epoch losses")
        
        # Create visualizations
        print(f"Creating visualizations in {viz_dir}/")
        create_visualizations(data, viz_dir)
        print("All visualizations saved successfully!")
        
        print("\nGenerated plots:")
        print("  - train_batch_loss.png: Training loss per batch")
        print("  - val_batch_loss.png: Validation loss per batch")
        print("  - train_epoch_loss.png: Training loss per epoch")
        print("  - val_epoch_loss.png: Validation loss per epoch")
        print("  - learning_rate.png: Learning rate schedule")
        print("  - gradient_norm.png: Gradient norm over training")
        print("  - train_3d_errors.png: Training 3D errors")
        print("  - val_3d_errors.png: Validation 3D errors")
        print("  - train_motion_errors.png: Training motion errors")
        print("  - val_motion_errors.png: Validation motion errors")
        
        # Print summary of available metrics
        print("\nAvailable metrics:")
        print(f"  Training metrics: {list(data['train_metrics'].keys())}")
        print(f"  Validation metrics: {list(data['val_metrics'].keys())}")
        
    except FileNotFoundError:
        print(f"Error: Could not find log file at {log_file_path}")
        print("Please make sure the file exists and update the log_file_path variable")
    except Exception as e:
        print(f"Error processing logs: {str(e)}")

if __name__ == "__main__":
    main()