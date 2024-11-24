import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def calculate_metrics(test_data):
    """
    Calculate accuracy, F1 score, and recall for Benchmark and Hubs rank.

    :param test_data: DataFrame containing the test results
    :return: Dictionary with accuracy, F1, and recall for both benchmark and hubs ranks
    """
    metrics = {'accuracy_bench': accuracy_score(test_data['answer_index'], test_data['bench_rank']),
               'accuracy_hubs': accuracy_score(test_data['answer_index'], test_data['hubs_rank']),
               'f1_bench': f1_score(test_data['answer_index'], test_data['bench_rank'], average='weighted',
                                    labels=np.unique(test_data['answer_index'])),
               'f1_hubs': f1_score(test_data['answer_index'], test_data['hubs_rank'], average='weighted',
                                   labels=np.unique(test_data['answer_index'])),
               'recall_bench': recall_score(test_data['answer_index'], test_data['bench_rank'], average='weighted',
                                            labels=np.unique(test_data['answer_index'])),
               'recall_hubs': recall_score(test_data['answer_index'], test_data['hubs_rank'], average='weighted',
                                           labels=np.unique(test_data['answer_index']))}

    return metrics


def plot_metrics_comparison(metrics):
    """
    Create bar plots comparing the metrics for Benchmark and Hubs rank.

    :param metrics: Dictionary containing accuracy, F1 score, and recall for both ranks
    """
    metric_labels = ['Accuracy', 'F1 Score', 'Recall']
    bench_metrics = [metrics['accuracy_bench'], metrics['f1_bench'], metrics['recall_bench']]
    hubs_metrics = [metrics['accuracy_hubs'], metrics['f1_hubs'], metrics['recall_hubs']]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].bar(metric_labels, bench_metrics, color='blue', alpha=0.7, label='Benchmark Rank')
    ax[0].bar(metric_labels, hubs_metrics, color='orange', alpha=0.7, label='Hubs Rank')
    ax[0].set_title("Comparison of Accuracy")
    ax[0].set_ylabel('Score')
    ax[0].legend()

    ax[1].bar(metric_labels, bench_metrics, color='blue', alpha=0.7, label='Benchmark Rank')
    ax[1].bar(metric_labels, hubs_metrics, color='orange', alpha=0.7, label='Hubs Rank')
    ax[1].set_title("Comparison of F1 Score")
    ax[1].set_ylabel('Score')
    ax[1].legend()

    ax[2].bar(metric_labels, bench_metrics, color='blue', alpha=0.7, label='Benchmark Rank')
    ax[2].bar(metric_labels, hubs_metrics, color='orange', alpha=0.7, label='Hubs Rank')
    ax[2].set_title("Comparison of Recall")
    ax[2].set_ylabel('Score')
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(test_data):
    """
    Plot confusion matrices for Benchmark Rank and Hubs Rank.

    :param test_data: DataFrame containing the test results
    """

    cm_bench = confusion_matrix(test_data['answer_index'], test_data['bench_rank'])
    cm_hubs = confusion_matrix(test_data['answer_index'], test_data['hubs_rank'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    sns.heatmap(cm_bench, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_data['answer_index']),
                yticklabels=np.unique(test_data['answer_index']), ax=ax1)
    ax1.set_title("Confusion Matrix: Benchmark Rank")
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(cm_hubs, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_data['answer_index']),
                yticklabels=np.unique(test_data['answer_index']), ax=ax2)
    ax2.set_title("Confusion Matrix: Hubs Rank")
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.show()


def generate_summary_plots(test_data):
    """
    Generate summary plots comparing Benchmark and Hubs Rank metrics.
    :param test_data: DataFrame containing the test results
    """

    metrics = calculate_metrics(test_data)
    plot_metrics_comparison(metrics)
    plot_confusion_matrices(test_data)
