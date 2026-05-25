# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Evaluation Module
# ============================================================================
# This module generates comprehensive evaluation metrics and visualizations
# as described in Chapter 6 (Experimental Results Analysis):
#   - Classification accuracy, precision, recall, F1-score
#   - Confusion matrix heatmap (Figure 6.5)
#   - Training vs Validation accuracy graph (Figure 6.2)
#   - Training vs Validation loss graph (Figure 6.3)
#   - Model performance evaluation metrics bar chart (Figure 6.4)
# ============================================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from . import config


def evaluate_model(model, test_generator):
    """
    Run model predictions on the test set and compute all evaluation metrics.
    
    As described in Chapter 6.3:
    - Accuracy: ~89-91%
    - Precision: ~87%
    - Recall: ~85-90%
    - F1 Score: ~86-88%
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
    
    Returns:
        dict: Dictionary containing all evaluation metrics and predictions
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    
    # Reset generator to ensure we start from the beginning
    test_generator.reset()
    
    # Get predictions for the entire test set
    print("[INFO] Generating predictions on test set...")
    predictions = model.predict(test_generator, verbose=1)
    
    # Get predicted class indices
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Ensure we have matching lengths
    y_pred = y_pred[:len(y_true)]
    
    # Get class names in order
    class_labels = sorted(config.CLASS_TO_INDEX.keys())
    
    # -----------------------------------------------------------------------
    # Compute evaluation metrics
    # -----------------------------------------------------------------------
    print("\n[INFO] Computing evaluation metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1-score (weighted average for imbalanced classes)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Classification report (per-class metrics)
    report = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision:         {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:            {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1-Score:          {f1:.4f} ({f1*100:.1f}%)")
    print(f"{'='*50}")
    
    # Print per-class classification report
    print("\n  Per-Class Classification Report:")
    print("-" * 60)
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_labels
    )
    print(report_str)
    
    # Collect results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': predictions,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_labels': class_labels
    }
    
    return results


def plot_training_history(history, save_dir=config.STATIC_IMAGES_DIR):
    """
    Plot training and validation accuracy/loss graphs.
    
    Generates:
    - Figure 6.2: Training and Validation Accuracy Graph
    - Figure 6.3: Training and Validation Loss Graph
    
    Args:
        history (dict): Training history dictionary
        save_dir (str): Directory to save plot images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # -------------------------------------------------------------------
    # Figure 6.2: Training and Validation Accuracy Graph
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['accuracy'], 'b-o',
            label='Training Accuracy', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_accuracy'], 'r-o',
            label='Validation Accuracy', linewidth=2, markersize=4)
    
    ax.set_title('Training and Validation Accuracy',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add phase boundary line
    phase1_epochs = config.EPOCHS_PHASE1
    if phase1_epochs < len(epochs):
        ax.axvline(x=phase1_epochs, color='green', linestyle='--',
                   alpha=0.7, label=f'Fine-tuning starts (Epoch {phase1_epochs})')
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    accuracy_path = os.path.join(save_dir, 'training_validation_accuracy.png')
    fig.savefig(accuracy_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] Accuracy graph: {accuracy_path}")
    
    # -------------------------------------------------------------------
    # Figure 6.3: Training and Validation Loss Graph
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['loss'], 'b-o',
            label='Training Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-o',
            label='Validation Loss', linewidth=2, markersize=4)
    
    ax.set_title('Training and Validation Loss',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add phase boundary line
    if phase1_epochs < len(epochs):
        ax.axvline(x=phase1_epochs, color='green', linestyle='--',
                   alpha=0.7, label=f'Fine-tuning starts (Epoch {phase1_epochs})')
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'training_validation_loss.png')
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] Loss graph: {loss_path}")


def plot_confusion_matrix(cm, class_labels,
                          save_dir=config.STATIC_IMAGES_DIR):
    """
    Generate and save a styled confusion matrix heatmap.
    
    Generates Figure 6.5: Confusion Matrix of Skin Lesion Classification Model
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_labels (list): List of class label names
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in cells
        fmt='d',              # Integer format
        cmap='Blues',          # Blue color palette
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_title('Confusion Matrix — Skin Lesion Classification',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] Confusion matrix: {cm_path}")


def plot_performance_metrics(results, save_dir=config.STATIC_IMAGES_DIR):
    """
    Generate a bar chart of model performance evaluation metrics.
    
    Generates Figure 6.4: Model Performance Evaluation Metrics
    
    Args:
        results (dict): Evaluation results dictionary
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Metrics to plot
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1_score']
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        metrics.keys(),
        [v * 100 for v in metrics.values()],
        color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'],
        edgecolor='white',
        linewidth=1.5,
        width=0.6
    )
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{value*100:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_title('Model Performance Evaluation Metrics',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_ylim([0, 105])
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_path = os.path.join(save_dir, 'performance_metrics.png')
    fig.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] Performance metrics: {metrics_path}")


def plot_class_distribution(save_dir=config.STATIC_IMAGES_DIR):
    """
    Plot the HAM10000 class distribution bar chart.
    Useful for the notebook and report.
    
    Args:
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    classes = list(config.CLASS_DISTRIBUTION.keys())
    counts = list(config.CLASS_DISTRIBUTION.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E91E63', '#9C27B0', '#2196F3', '#00BCD4',
              '#FF5722', '#4CAF50', '#FF9800']
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    ax.set_title('HAM10000 Dataset — Class Distribution',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Skin Lesion Class', fontsize=13)
    ax.set_ylabel('Number of Images', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'class_distribution.png')
    fig.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] Class distribution: {dist_path}")


def generate_all_evaluation_plots(model, test_generator, history=None):
    """
    Generate all evaluation plots and metrics in one call.
    
    This function produces all figures referenced in Chapter 6:
    - Figure 6.2: Training/Validation Accuracy
    - Figure 6.3: Training/Validation Loss
    - Figure 6.4: Performance Metrics
    - Figure 6.5: Confusion Matrix
    
    Args:
        model (tf.keras.Model): Trained model
        test_generator: Test data generator
        history (dict): Training history. If None, loads from file.
    
    Returns:
        dict: Complete evaluation results
    """
    print("\n" + "=" * 60)
    print("  GENERATING EVALUATION PLOTS")
    print("=" * 60)
    
    # Evaluate model on test set
    results = evaluate_model(model, test_generator)
    
    # Generate confusion matrix plot
    print("\n[INFO] Generating confusion matrix plot...")
    plot_confusion_matrix(results['confusion_matrix'],
                          results['class_labels'])
    
    # Generate performance metrics bar chart
    print("[INFO] Generating performance metrics plot...")
    plot_performance_metrics(results)
    
    # Generate class distribution plot
    print("[INFO] Generating class distribution plot...")
    plot_class_distribution()
    
    # Generate training history plots if history is available
    if history is not None:
        print("[INFO] Generating training history plots...")
        plot_training_history(history)
    else:
        # Try to load from saved file
        history_path = os.path.join(config.MODELS_DIR,
                                     'training_history.json')
        if os.path.exists(history_path):
            print("[INFO] Loading training history from file...")
            with open(history_path, 'r') as f:
                history = json.load(f)
            plot_training_history(history)
        else:
            print("  [WARNING] No training history available for plotting")
    
    # Save evaluation results as JSON
    results_json = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'classification_report': results['classification_report']
    }
    
    results_path = os.path.join(config.STATIC_IMAGES_DIR,
                                 'evaluation_results.json')
    os.makedirs(config.STATIC_IMAGES_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  [SAVED] Evaluation results: {results_path}")
    
    print("\n" + "=" * 60)
    print("  ALL EVALUATION PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    
    return results
