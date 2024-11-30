import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def get_tagged_results(path='tagged_tests.csv'):
    def calculate_metrics(fn, tn, tp, fp):
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity,
            'FPR': fpr,
            'FNR': fnr,
            'Balanced Accuracy': balanced_accuracy
        }

    final_df = pd.read_csv(path)
    hubs_values = final_df["hubs_rank"].value_counts()
    bench_values = final_df["bench_rank"].value_counts()

    fn_hubs = final_df[(final_df["bench_rank"] == 'Correct') & (final_df["hubs_rank"] == "Don't know")]
    tn_hubs = final_df[(final_df["bench_rank"] != 'Correct') & (final_df["hubs_rank"] == "Don't know")]
    tp_hubs = final_df[(final_df["hubs_rank"] == "Correct")]
    fp_hubs = final_df[(final_df["hubs_rank"] == "Incorrect")]

    fn_bench = final_df[(final_df["hubs_rank"] == 'Correct') & (final_df["bench_rank"] == "Don't know")]
    tn_bench = final_df[(final_df["hubs_rank"] != 'Correct') & (final_df["bench_rank"] == "Don't know")]
    tp_bench = final_df[(final_df["bench_rank"] == "Correct")]
    fp_bench = final_df[(final_df["bench_rank"] == "Incorrect")]

    metrics_hubs = calculate_metrics(len(fn_hubs), len(tn_hubs), len(tp_hubs), len(fp_hubs))
    metrics_bench = calculate_metrics(len(fn_bench), len(tn_bench), len(tp_bench), len(fp_bench))
    metrics_data = {
        'Hubs Rank': list(metrics_hubs.values()),
        'Benchmark Rank': list(metrics_bench.values())
    }

    metrics_df = pd.DataFrame(metrics_data, index=metrics_hubs.keys())
    metrics_df = metrics_df.rename({"Hubs Rank": "Selective RAG & HITS Reranking", "Benchmark Rank": "Basic RAG"},
                                   axis=1)
    print("\nModels Comparison:")
    print(metrics_df)
    return final_df, metrics_df


def plot_metrics_comparison(df):
    width = 0.35
    x = np.arange(len(df))
    metrics = df.index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))

    hubs_bars = ax.bar(x - width / 2, df['Selective RAG & HITS Reranking'], width,
                       label='Selective RAG & HITS Reranking', color='#387F39', edgecolor="#387F39", linewidth=1.25,
                       alpha=0.75)
    bench_bars = ax.bar(x * 1.001 + width / 2, df['Basic RAG'], width, label='Basic RAG', color='#96C9F4',
                        linewidth=1.25,
                        edgecolor="#96C9F4", alpha=0.75)

    ax.set_xlabel('\nMetrics')
    ax.set_ylabel("")
    ax.set_title('Evaluation Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, ha='center', fontweight='bold')

    ax.legend(loc='upper left')

    for bars in [hubs_bars, bench_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height * 0.99),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_results_table_T(df):
    """
    Plot confusion matrices for Benchmark Rank and Hubs Rank.

    :param df: DataFrame containing the test results
    """

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['#A7D397'] * len(df.T.columns)
    formatted_values = [[f"{value:.2f}" for value in row] for row in df.T.values]

    table = ax.table(cellText=formatted_values, colLabels=df.T.columns, rowLabels=df.T.index,
                     colColours=colors, loc='center', cellLoc='center')

    ax.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    table.set_fontweight = 'bold'
    table.auto_set_column_width([0, 1])
    plt.tight_layout()
    plt.show()


def plot_results_table(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#A7D397', '#96C9F4']
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, colColours=colors, loc='center',
                     cellLoc='center')

    ax.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    table.auto_set_column_width([0, 1])
    plt.tight_layout()
    plt.show()


def generate_summary():
    """
    Generate summary plots comparing Benchmark and Hubs Rank metrics.
    """
    tagged_results_df, metrics_df = get_tagged_results()
    plot_metrics_comparison(metrics_df)
    plot_results_table_T(metrics_df)
