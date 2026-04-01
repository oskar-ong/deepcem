from datetime import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DB_PATH = "cem_results.db"
OUTPUT_PLOT = "er_performance_iterations.png"
BACKUP_FILENAME = f"experiment_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def save_to_csv(df, output_path):
    """Saves the full merged dataset to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Backup created: {output_path}")
    except Exception as e:
        print(f"Failed to save CSV backup: {e}")


def fetch_experiment_data(db_path):
    """Joins runs and metrics into a single DataFrame."""
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found.")
        return None

    conn = sqlite3.connect(db_path)
    # Join on run_id to get hyperparameters alongside metrics
    query = """
    SELECT 
        r.dataset, r.lm, r.neg_ratio,
        m.pollution, m.iteration, m.entity, m.testset_type,
        m.precision, m.recall, m.f1_score, m.runtime
    FROM metrics m
    JOIN runs r ON m.run_id = r.run_id
    ORDER BY m.pollution, m.iteration
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def print_markdown_report(df):
    """Prints a human-readable Markdown table for easy copy-pasting."""
    print("\n" + "="*30)
    print("EXPERIMENT SUMMARY REPORT")
    print("="*30 + "\n")

    # Grouping by the most important factors and showing the latest iteration
    # To see the final results of the Collective ER process
    summary = df.sort_values('iteration').groupby(
        ['dataset', 'pollution', 'entity']).tail(1)

    # format columns for readability
    report_df = summary[['dataset', 'pollution', 'entity',
                         'iteration', 'precision', 'recall', 'f1_score']]

    # Rounding for cleanliness
    report_df = report_df.round(4)

    # to_markdown requires the 'tabulate' library installed
    try:
        print(report_df.to_markdown(index=False))
    except ImportError:
        print(report_df.to_string(index=False))
        print("\n(Tip: Install 'tabulate' to get pretty markdown tables next time)")


def plot_performance_trends(df):
    """Generates a line plot showing F1-Score progress over iterations."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create the plot
    # x: iteration, y: F1, hue: pollution level, style: entity
    plot = sns.lineplot(
        data=df,
        x="iteration",
        y="f1_score",
        hue="pollution",
        style="entity",
        markers=True,
        dashes=False,
        linewidth=2.5
    )

    plt.title(
        f"Collective ER Performance: {df['dataset'].unique()[0]}", fontsize=15)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Pollution Level",
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\nPlot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    results_df = fetch_experiment_data(DB_PATH)

    if results_df is not None:
        print_markdown_report(results_df)
        plot_performance_trends(results_df)
        save_to_csv(results_df, BACKUP_FILENAME)
