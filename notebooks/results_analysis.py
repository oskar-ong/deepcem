from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

import pandas as pd


def run_sql_query(query: str) -> pd.DataFrame:
    conn = sqlite3.connect('../cem_results.db')
    try:

        # Execute and load into DataFrame
        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()
    return df


def run_sql_query_params(query: str, params=None) -> pd.DataFrame:
    conn = sqlite3.connect('../cem_results.db')
    try:

        # Execute and load into DataFrame
        df = pd.read_sql_query(query, conn, params=(params,))

    finally:
        conn.close()
    return df


def query_metrics(job_id):
    query = f"""SELECT 
        r.dataset,
        m.entity,
        m.metric_type,
        AVG(m.f1_score) AS mean_f1,
        SQRT((SUM(m.f1_score * m.f1_score) - (SUM(m.f1_score) * SUM(m.f1_score) / COUNT(m.f1_score))) / (COUNT(m.f1_score) - 1)) AS std_f1,
        COUNT(r.seed) AS num_seeds,
        r.train_size,
        r.pollution,
        r.batch_size,
        r.max_len,
        r.learning_rate,
        r.epochs,
        r.lm,
        r.neg_ratio
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE m.is_final = 1  AND m.run_id LIKE '{job_id}_%'
    GROUP BY 
        r.dataset, 
        r.train_size, 
        r.pollution, 
        r.batch_size, 
        r.max_len, 
        r.learning_rate, 
        r.epochs, 
        r.lm, 
        r.neg_ratio,
        m.entity, 
        m.metric_type
    ORDER BY r.dataset, m.entity, m.metric_type;"""

    df = run_sql_query(query)
    df.drop(columns=["batch_size", "max_len",
            "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


def query_metrics_p1(job_id):
    query = f"""SELECT 
        r.dataset,
        m.entity,
        m.metric_type,
        AVG(m.f1_score) AS mean_f1,
        SQRT((SUM(m.f1_score * m.f1_score) - (SUM(m.f1_score) * SUM(m.f1_score) / COUNT(m.f1_score))) / (COUNT(m.f1_score) - 1)) AS std_f1,
        COUNT(r.seed) AS num_seeds,
        r.train_size,
        r.pollution,
        r.batch_size,
        r.max_len,
        r.learning_rate,
        r.epochs,
        r.lm,
        r.neg_ratio
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE m.metric_type = 'phase1'  AND m.run_id LIKE '{job_id}_%'
    GROUP BY 
        r.dataset, 
        r.train_size, 
        r.pollution, 
        r.batch_size, 
        r.max_len, 
        r.learning_rate, 
        r.epochs, 
        r.lm, 
        r.neg_ratio,
        m.entity, 
        m.metric_type
    ORDER BY r.dataset, m.entity, m.metric_type;"""

    df = run_sql_query(query)
    df.drop(columns=["batch_size", "max_len",
            "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


def query_metrics_individual(job_id):
    query = f"""SELECT 
        r.dataset,
        m.entity,
        m.metric_type,
        m.f1_score, 
        r.seed,
        r.train_size,
        r.pollution,
        r.batch_size,
        r.max_len,
        r.learning_rate,
        r.epochs,
        r.lm,
        r.neg_ratio
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE m.is_final = 1  AND m.run_id LIKE '{job_id}_%'
    ORDER BY r.dataset, m.entity, m.metric_type;"""

    df = run_sql_query(query)
    df.drop(columns=["batch_size", "max_len",
            "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


def query_prelimary(job_id):
    query = f"""SELECT 
        r.dataset,
        m.entity,
        m.metric_type,
        AVG(m.f1_score) AS mean_f1,
        SQRT((SUM(m.f1_score * m.f1_score) - (SUM(m.f1_score) * SUM(m.f1_score) / COUNT(m.f1_score))) / (COUNT(m.f1_score) - 1)) AS std_f1,
        COUNT(r.seed) AS num_seeds,
        r.train_size,
        r.pollution,
        r.batch_size,
        r.max_len,
        r.learning_rate,
        r.epochs,
        r.lm,
        r.neg_ratio
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE m.metric_type IN ('phase1', 'phase2', 'baseline') AND m.run_id LIKE '{job_id}_%'
    GROUP BY 
        r.dataset, 
        r.train_size, 
        r.pollution, 
        r.batch_size, 
        r.max_len, 
        r.learning_rate, 
        r.epochs, 
        r.lm, 
        r.neg_ratio,
        m.entity, 
        m.metric_type
    ORDER BY r.dataset, m.entity, m.metric_type;"""

    df = run_sql_query(query)
    df.drop(columns=["batch_size", "max_len",
            "learning_rate", "lm", "neg_ratio"], inplace=True)

    # 1. Define the logical order
    pollution_order = ['source', 'low', 'medium', 'high']

    # 2. Apply the categorical type to the column
    df['pollution'] = pd.Categorical(
        df['pollution'],
        categories=pollution_order,
        ordered=True
    )

    # 3. Sort the DataFrame to reflect this order
    # This ensures that when you generate the table or plot, the rows are in order
    df = df.sort_values(by=['dataset', 'entity', 'pollution', 'metric_type'])
    return df


def query_retem(job_id):
    query = f"""SELECT * FROM metrics WHERE 
    is_final = 1 AND run_id = '{job_id}' AND metric_type = 'test';"""

    df = run_sql_query(query)
    # df.drop(columns=["batch_size", "max_len",
    #         "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


def query_baseline(job_id):
    query = f"""SELECT * FROM metrics WHERE 
    is_final = 1 AND run_id = '{job_id}' AND metric_type = 'baseline';"""

    df = run_sql_query(query)
    # df.drop(columns=["batch_size", "max_len",
    #         "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


def plot_retem_vs_base_old(jobid):

    query = """
    SELECT 
        t.run_id AS job_id,
        t.f1_score AS retem_f1,
        b.f1_score AS baseline_f1,
        r.pollution AS pollution
    FROM metrics t
    JOIN metrics b ON t.run_id = b.run_id
    JOIN runs r ON t.run_id = r.run_id
    WHERE
    t.run_id LIKE ? AND
    t.is_final = 1 
      AND t.metric_type = 'test'
      AND b.metric_type = 'baseline'
    """
    like_pattern = f"{jobid}_%"

    df = run_sql_query_params(query, like_pattern)

    if df.empty:
        print("No matching pairs found. Verify how 'test' and 'baseline' rows are linked in your schema.")
        return

    # 2. Calculate the difference for the Y-axis
    df['f1_diff'] = df['retem_f1'] - df['baseline_f1']

    pollution_colors = {
        'source': '#2ca02c',  # Clean green
        'low': '#1f77b4',     # Muted blue
        'medium': '#ff7f0e',  # Alert orange
        'high': '#d62728'     # Warning red
    }

    plt.figure(figsize=(10, 6))

    for level in ['source', 'low', 'medium', 'high']:
        sub_df = df[df['pollution'] == level]

        # Only plot if this specific pollution level exists in your current query slice
        if not sub_df.empty:
            plt.scatter(
                sub_df['baseline_f1'],
                sub_df['f1_diff'],
                label=level.capitalize(),       # Capitalizes 'low' to 'Low' for clean display
                s=40,       # Shrank from 130 to 40 so they take up less physical space
                alpha=0.5,  # Made 50% transparent so overlaps blend together
                facecolors='none',  # Makes the circles hollow
                edgecolors=pollution_colors[level],  # Colors the outer ring
                linewidths=1.5,
                zorder=3
            )

    # Red baseline reference line (Y=0)
    plt.axhline(0, color='red', linestyle='--',
                linewidth=1.5, alpha=0.6, zorder=2)

    plt.legend(
        title="Pollution Level",
        title_fontsize=11,
        fontsize=10,
        loc='upper right',
        frameon=True,
        shadow=False,
        facecolor='white',
        edgecolor='#ccc'
    )
    # 4. Label each point with a short snippet of its Job ID
    # for _, row in df.iterrows():
    #     job_label = str(row['job_id'])
    #     # If your job IDs are long UUID strings, slice them down to 8 characters so the plot isn't crowded
    #     if len(job_label) > 8:
    #         job_label = job_label[:8] + "..."

    #     plt.annotate(
    #         job_label,
    #         (row['retem_f1'], row['f1_diff']),
    #         textcoords="offset points",
    #         xytext=(0, 8),
    #         ha='center',
    #         fontsize=8.5,
    #         fontweight='semibold',
    #         alpha=0.8
    #     )

    # Aesthetics and scaling
    plt.title('F1 Score difference: RETEM vs. Baseline',
              fontsize=13, pad=15, fontweight='bold')
    plt.xlabel('Baseline F1 Score', fontsize=11, labelpad=10)
    plt.ylabel('F1 Score Improvement (RETEM - Baseline)',
               fontsize=11, labelpad=10)

    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # Set reasonable plot padding
    plt.xlim(df['retem_f1'].min() - 0.05,
             min(df['retem_f1'].max() + 0.05, 1.02))

    plt.tight_layout()
    plt.show()


def plot_retem_vs_base(jobid, title):
    # 1. Fetch data safely
    query = """
    SELECT 
        t.run_id AS job_id,
        t.f1_score AS retem_f1,
        b.f1_score AS baseline_f1,
        r.pollution AS pollution
    FROM metrics t
    JOIN metrics b ON t.run_id = b.run_id
    JOIN runs r    ON t.run_id = r.run_id
    WHERE t.run_id LIKE ? 
      AND t.is_final = 1 
      AND t.metric_type = 'test'
      AND b.metric_type = 'baseline'
    """
    like_pattern = f"{jobid}_%"
    df = run_sql_query_params(query, like_pattern)

    if df.empty:
        print(f"No matching pairs found for pattern: {like_pattern}")
        return

    # Calculate the performance delta (Y-axis)
    df['f1_diff'] = df['retem_f1'] - df['baseline_f1']

    # Setup Academic Styling & Color Profile
    sns.set_theme(style="whitegrid")

    pollution_colors = {
        'source': '#2ca02c',  # Clean green
        'low': '#1f77b4',     # Muted blue
        'medium': '#ff7f0e',  # Alert orange
        'high': '#d62728'     # Warning red
    }
    logical_order = ['source', 'low', 'medium', 'high']
    X_AXIS_METRIC = 'baseline_f1'

    if X_AXIS_METRIC == 'baseline_f1':
        leg_pos = "upper right"
    else:
        leg_pos = "upper_left"

    # =========================================================================
    # FIXED: Initialize JointGrid BARE (No global hue variable)
    # =========================================================================
    g = sns.JointGrid(
        data=df,
        x=X_AXIS_METRIC,
        y="f1_diff",
        height=7
    )

    # =========================================================================
    # FIXED: Plot scatter dots onto the center axis manually.
    # This preserves color data while maintaining hollow circles.
    # =========================================================================
    for level in logical_order:
        sub_df = df[df['pollution'] == level]
        if not sub_df.empty:
            g.ax_joint.scatter(
                sub_df[X_AXIS_METRIC],
                sub_df['f1_diff'],
                label=level.capitalize(),
                s=55,
                alpha=0.7,
                facecolors='none',       # Hollow circles to handle overplotting
                edgecolors=pollution_colors[level],
                linewidths=1.5,
                zorder=3
            )

    # 3. Draw clean, stacked marginal histograms manually on the side axes
    sns.histplot(
        data=df, x=X_AXIS_METRIC, ax=g.ax_marg_x, hue="pollution",
        palette=pollution_colors, hue_order=logical_order,
        multiple="stack", element="step", alpha=0.35, legend=False
    )
    sns.histplot(
        data=df, y="f1_diff", ax=g.ax_marg_y, hue="pollution",
        palette=pollution_colors, hue_order=logical_order,
        multiple="stack", element="step", alpha=0.35, legend=False
    )

    # Add Red Baseline Reference Line (Y=0)
    g.ax_joint.axhline(0, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.6, zorder=2)

    # Polish Labels & Typography
    x_label_text = 'RETEM F1 Score (Absolute)' if X_AXIS_METRIC == 'retem_f1' else 'Baseline F1 Score'
    g.set_axis_labels(
        x_label_text, 'F1 Score Improvement (RETEM - Baseline)', fontsize=11, labelpad=10)
    g.fig.suptitle(
        f"Difference F1 Score: Baseline vs RETEM - {title}", y=1.03, fontweight='bold', fontsize=13)

    # 5. Generate the crisp, discrete legend
    leg = g.ax_joint.legend(
        title="Pollution Level",
        title_fontsize=11,
        fontsize=10,
        loc=leg_pos,
        frameon=True,
        facecolor='white',
        edgecolor='#ccc'
    )

    # =========================================================================
    # THESIS POLISH: Force legend icons to match the plot perfectly (Hollow)
    # =========================================================================
    try:
        handles = leg.legend_handles if hasattr(
            leg, 'legend_handles') else leg.legendHandles
        for handle in handles:
            handle.set_facecolor('none')   # Keep the inside hollow
            # Beef up the outline slightly so it's readable
            handle.set_linewidth(2.0)
            # Make the outline 100% opaque in the legend
            handle.set_alpha(1.0)
    except Exception:
        pass
    # =========================================================================

    # Pad boundaries dynamically so dots near 0.0 or 1.0 don't get sliced in half
    g.ax_joint.set_xlim(df[X_AXIS_METRIC].min() - 0.04,
                        min(df[X_AXIS_METRIC].max() + 0.04, 1.02))

    plt.show()
    return df

# Example usage:
# plot_all_job_performances('your_database.db')


def generate_comparison_latex(df, test_metric='test', dataset_name='music'):
    # 1. Pivot the dataframe to get Baseline and Test side-by-side
    # We index by entity and pollution to keep them as rows

    # 1. Define the logical order
    pollution_order = ['source', 'low', 'medium', 'high']

    # 2. Apply the categorical type to the column
    df['pollution'] = pd.Categorical(
        df['pollution'],
        categories=pollution_order,
        ordered=True
    )

    # 3. Sort the DataFrame to reflect this order
    # This ensures that when you generate the table or plot, the rows are in order
    df = df.sort_values(by=['dataset', 'entity', 'pollution', 'metric_type'])

    pivot_df = df.pivot_table(
        index=['entity', 'pollution'],
        columns='metric_type',
        values=['mean_f1', 'std_f1']
    )

    # 2. Filter for only the levels we need (Baseline and the specific Test metric)
    # This ensures we don't have extra columns if the DF contains Phase 1, Phase 2, etc.
    try:
        baseline_mean = pivot_df[('mean_f1', 'baseline')]
        baseline_std = pivot_df[('std_f1', 'baseline')]
        test_mean = pivot_df[('mean_f1', test_metric)]
        test_std = pivot_df[('std_f1', test_metric)]
    except KeyError:
        return "Error: Metric types 'baseline' or '{}' not found in DataFrame.".format(test_metric)

    # 3. Build the LaTeX string
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Main Experiment: Baseline vs. RETEM - Dataset {dataset_name}}}",
        "\\label{tab:main_exp_}}",
        "\\begin{tabular}{ll c cc c cc}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{\\textbf{Baseline}} & & \\multicolumn{2}{c}{\\textbf{RETEM}} \\\\",
        "\\cmidrule{4-5} \\cmidrule{7-8}",
        "\\textbf{Entity} & \\textbf{Pollution} & & \\textbf{Mean F1} & \\textbf{SD} & & \\textbf{Mean F1} & \\textbf{SD} \\\\",
        "\\midrule"
    ]

    # 4. Iterate through the pivoted rows and apply bolding logic
    for (entity, pollution) in pivot_df.index:
        m_b = baseline_mean.loc[(entity, pollution)]
        s_b = baseline_std.loc[(entity, pollution)]
        m_t = test_mean.loc[(entity, pollution)]
        s_t = test_std.loc[(entity, pollution)]

        # Bolding logic for the higher mean
        val_b = f"\\textbf{{{m_b:.4f}}}" if m_b > m_t else f"{m_b:.4f}"
        val_t = f"\\textbf{{{m_t:.4f}}}" if m_t > m_b else f"{m_t:.4f}"

        row = f"{entity} & {pollution} & & {val_b} & {s_b:.4f} & & {val_t} & {s_t:.4f} \\\\"
        latex_lines.append(row)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def generate_comparison_latex_prelim(df, dataset_name='music'):
    # 1. Pivot the dataframe to get all metric types side-by-side
    pivot_df = df.pivot_table(
        index=['entity', 'pollution'],
        columns='metric_type',
        values=['mean_f1', 'std_f1']
    )

    # 2. Hardcode the metrics we want to extract
    target_phases = ['baseline', 'phase1', 'phase2']

    try:
        # Extract means and stds for all three phases
        m_b, s_b = pivot_df[('mean_f1', 'baseline')
                            ], pivot_df[('std_f1', 'baseline')]
        m_p1, s_p1 = pivot_df[('mean_f1', 'phase1')
                              ], pivot_df[('std_f1', 'phase1')]
        m_p2, s_p2 = pivot_df[('mean_f1', 'phase2')
                              ], pivot_df[('std_f1', 'phase2')]
    except KeyError as e:
        return f"Error: Required metric type {e} not found in DataFrame."

    # 3. Build the LaTeX string
    # tabular columns: Entity(l), Pollution(l), separator(c), B_Mean(c), B_SD(c), sep, P1_Mean, P1_SD, sep, P2_Mean, P2_SD
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Baseline (Ditto pure) vs. Phase 1 (Empty Relational Scores) vs. Phase 2 (Relational Scores based on Ground-Truth) - Dataset {dataset_name}}}",
        "\\label{tab:prelim_exp_}",
        "\\begin{tabular}{ll c cc c cc c cc}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{\\textbf{Baseline}} & & \\multicolumn{2}{c}{\\textbf{Phase 1}} & & \\multicolumn{2}{c}{\\textbf{Phase 2}} \\\\",
        "\\cmidrule{4-5} \\cmidrule{7-8} \\cmidrule{10-11}",
        "\\textbf{Entity} & \\textbf{Pollution} & & \\textbf{Mean} & \\textbf{SD} & & \\textbf{Mean} & \\textbf{SD} & & \\textbf{Mean} & \\textbf{SD} \\\\",
        "\\midrule"
    ]

    # 4. Iterate through the rows and apply bolding logic across the three phases
    for (entity, pollution) in pivot_df.index:
        # Current row values
        vals_mean = {
            'b': m_b.loc[(entity, pollution)],
            'p1': m_p1.loc[(entity, pollution)],
            'p2': m_p2.loc[(entity, pollution)]
        }
        vals_std = {
            'b': s_b.loc[(entity, pollution)],
            'p1': s_p1.loc[(entity, pollution)],
            'p2': s_p2.loc[(entity, pollution)]
        }

        # Determine the maximum mean to apply bolding
        max_mean = max(vals_mean.values())

        def format_bold(val, target_max):
            return f"\\textbf{{{val:.4f}}}" if val == target_max else f"{val:.4f}"

        # Format strings for the row
        b_str = f"{format_bold(vals_mean['b'], max_mean)} & {vals_std['b']:.4f}"
        p1_str = f"{format_bold(vals_mean['p1'], max_mean)} & {vals_std['p1']:.4f}"
        p2_str = f"{format_bold(vals_mean['p2'], max_mean)} & {vals_std['p2']:.4f}"

        row = f"{entity} & {pollution} & & {b_str} & & {p1_str} & & {p2_str} \\\\"
        latex_lines.append(row)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def plot_degradation_line(df, dataset, rtype, lm="roberta"):
    # 1. Re-sort pollution for correct plotting
    pollution_order = ['source', 'low', 'medium', 'high']
    df['pollution'] = pd.Categorical(
        df['pollution'], categories=pollution_order, ordered=True)

# 1. Get unique entities from your dataframe
    entities = df['entity'].unique()

    # 2. Iterate through each entity to create individual plots
    for entity in entities:
        # Filter data for the specific entity
        entity_df = df[df['entity'] == entity].sort_values("pollution")

        # Initialize a new figure for each plot
        plt.figure(figsize=(8, 5))
        sns.set_theme(style="whitegrid")
        ax = plt.gca()

        # Define color palette to match previous plots
        palette = sns.color_palette(
            "tab10", n_colors=len(df['metric_type'].unique()))
        metrics = entity_df['metric_type'].unique()

        # 3. Plot lines and SD bands for each metric type (Baseline, Phase 1, Phase 2, etc.)
        for i, metric in enumerate(metrics):
            subset = entity_df[entity_df['metric_type'] == metric]
            color = palette[i]

            # Plot the line
            sns.lineplot(
                data=subset,
                x="pollution",
                y="mean_f1",
                marker="o",
                label=metric,
                color=color
            )
            # Edit here: Use Barplot instead
            # sns.barplot(
            #     data=subset,
            #     x="pollution",
            #     y="mean_f1",
            #     # marker="o",
            #     label=metric,
            #     color=color
            # )

            # Plot the shaded SD band (Mean +/- SD)
            # Mapping categorical x-axis to numeric positions for fill_between
            x_coords = range(len(subset['pollution']))
            ax.fill_between(
                x_coords,
                subset['mean_f1'] - subset['std_f1'],
                subset['mean_f1'] + subset['std_f1'],
                alpha=0.2,
                color=color
            )

        # 4. Final formatting for the individual plot
        plt.ylim(0, 1)  # Fix Y-axis from 0 to 1
        plt.title(f"Entity: {entity}")
        plt.xlabel("Pollution Level")
        plt.ylabel("Mean F1-Score")
        plt.legend(title="Metric Type", loc='lower left')
        plt.tight_layout()

        # 5. Save or Show the individual plot
        out_path = Path(
            f"../img/{lm}/{dataset}/{rtype}")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../img/{lm}/{dataset}/{rtype}/f1_score_{entity}.png")
        plt.show()


def plot_degradation_barplot(df, dataset, rtype, lm="roberta"):
    pollution_order = ['source', 'low', 'medium', 'high']
    df['pollution'] = pd.Categorical(
        df['pollution'], categories=pollution_order, ordered=True)

    # cast to string
    df['metric_type'] = df['metric_type'].astype(str)
    # unique metric types: baseline, phase1, phase2, test
    metric_types = df['metric_type'].unique()

    entities = df['entity'].unique()

    for entity in entities:
        entity_df = df[df['entity'] == entity].sort_values("pollution")

        valid_data = entity_df.dropna(subset=['mean_f1', 'std_f1'])
        if valid_data.empty:
            continue

        min_score = (entity_df['mean_f1'] - valid_data['std_f1']).min()
        lower_limit = min_score - 0.05

        plt.figure(figsize=(8, 5))
        sns.set_theme(style="whitegrid")

        ax = sns.barplot(
            data=entity_df,
            x="pollution",
            y="mean_f1",
            hue="metric_type",
            palette="tab10"
        )

        for i, metric_label in enumerate(metric_types):
            if i >= len(ax.containers):
                break

            container = ax.containers[i]

            subset = entity_df[entity_df['metric_type']
                               == metric_label].sort_values("pollution")

            if subset.empty:
                continue

            # ax.bar_label(
            #     container,
            #     fmt='%.3f',
            #     padding=-15,
            #     fontsize=9,
            #     fontweight='bold'
            # )

            for bar, val in zip(container, subset['mean_f1'].values):
                # Calculate X (center of the bar)
                x_pos = bar.get_x() + bar.get_width() / 2

                # Position at the base: lower_limit + a tiny buffer
                # va='bottom' ensures the text sits ON TOP of this coordinate
                ax.text(
                    x=x_pos,
                    y=lower_limit + 0.01,
                    s=f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='white'   # Use white so it's readable against the bar color
                    # rotation=90      # Vertical labels look cleaner at the bottom
                )
            # metric_label = container.get_label()
            # print(metric_label)
            # subset = entity_df[entity_df['metric_type']
            #                    == metric_label].sort_values("pollution")

            x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]

            y_stds = subset['std_f1'].values
            # print(subset['std_f1'].values)
            y_means = subset['mean_f1'].values

            lower_err = y_stds
            upper_err = []

            for m, s in zip(y_means, y_stds):
                if (m + s) > 1.0:
                    upper_err.append(1.0 - m)  # Clamp to the ceiling of 1.0
                else:
                    upper_err.append(s)

            asymmetric_err = [lower_err, upper_err]

            # Only plot if we have matching data (prevents size mismatch errors)
            if len(x_coords) == len(y_stds):
                ax.errorbar(
                    x=x_coords,
                    y=y_means,
                    yerr=asymmetric_err,
                    fmt='none',  # 'none' means don't connect with a line
                    c='black',
                    capsize=4,   # Width of the horizontal "caps"
                    elinewidth=1.2,
                    alpha=0.8
                )

        # 3. Final formatting
        plt.ylim(lower_limit, 1)
        plt.title(f"Entity: {entity}")
        plt.xlabel("Pollution Level")
        plt.ylabel("Mean F1-Score")

        # Move legend outside if it's crowded
        plt.legend(title="Metric Type", loc='upper left',
                   bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # 4. Save/Show
        out_path = Path(f"../img/barplots/{lm}/{dataset}/{rtype}")
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / f"f1_score_{entity}.png")
        plt.show()


# Boxplot
def plot_degradation_boxplot(df, dataset, rtype, lm="roberta", y_min=0.5):
    pollution_order = ['source', 'low', 'medium', 'high']
    df['pollution'] = pd.Categorical(
        df['pollution'], categories=pollution_order, ordered=True)

    entities = df['entity'].unique()
    metric_types = df['metric_type'].unique()
    n_metrics = len(metric_types)

    for entity in entities:
        entity_df = df[df['entity'] == entity].sort_values("pollution")

        min_score = entity_df['f1_score'].min()
        lower_limit = min_score - 0.1

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # 2. Plot the Boxplot
        # Note: 'y' should be your raw score column (e.g., 'f1_score')
        # If you only have 'mean_f1', this will just show a line.
        ax = sns.boxplot(
            data=entity_df,
            x="pollution",
            y="f1_score",  # Use the raw score column here
            hue="metric_type",
            palette="tab10",
            # showmeans=True,
            # meanprops={
            #     "marker": "d",
            #     "markerfacecolor": "white",
            #     "markeredgecolor": "black",
            #     "markersize": 5
            # },
            fliersize=4,      # Size of outlier points
            linewidth=1.5
        )

        for i, p_level in enumerate(pollution_order):
            for j, m_type in enumerate(metric_types):
                # Filter for this specific box
                subset = entity_df[(entity_df['pollution'] == p_level) &
                                   (entity_df['metric_type'] == m_type)]

                if not subset.empty:
                    # Calculate mean
                    mean_val = subset['f1_score'].mean()

                    # Calculate X-coordinate:
                    # i is the category center (0, 1, 2...)
                    # The second part calculates the shift based on hue index
                    width = 0.8  # Default seaborn box width
                    x_pos = i + (j - (n_metrics - 1) / 2) * (width / n_metrics)

                    # Add text slightly above the mean marker
                    ax.text(
                        x_pos,
                        mean_val + 0.02,
                        f'{mean_val:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='black'
                    )

        # Optional: Add a swarmplot on top to show individual data points
        # sns.stripplot(
        #     data=entity_df,
        #     x="pollution",
        #     y="f1_score",
        #     hue="metric_type",
        #     dodge=True,
        #     alpha=0.3,
        #     palette="dark:black"
        # )

        # 3. Final formatting
        plt.ylim(lower_limit, 1.05)
        plt.title(f"Score Distribution - Entity: {entity}", pad=20)
        plt.xlabel("Pollution Level")
        plt.ylabel("F1-Score")

        # Handle legend (preventing duplicates if using stripplot)
        handles, labels = ax.get_legend_handles_labels()
        # If you have 3 metric types, just take the first 3 handles
        unique_labels = len(df['metric_type'].unique())
        plt.legend(
            handles[:unique_labels],
            labels[:unique_labels],
            title="Metric Type",
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )

        plt.tight_layout()

        # 4. Save/Show
        out_path = Path(f"../img/boxplots/{lm}/{dataset}/{rtype}")
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / f"boxplot_f1_{entity}.png")
        plt.show()


def investigate_outliers(outlier_row, job_id):
    """
    Fetches raw scores for all seeds associated with a high-SD configuration.
    """
    # Extract row identifiers
    dataset = outlier_row['dataset']
    entity = outlier_row['entity']
    m_type = outlier_row['metric_type']
    pollution = outlier_row['pollution']

    # Query individual runs
    query = f"""
    SELECT 
        m.run_id,
        m.entity,
        r.seed,
        m.f1_score,
        m.precision,
        m.recall,
        m.iteration
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    WHERE r.dataset = '{dataset}'
      AND m.entity = '{entity}'
      AND m.metric_type = '{m_type}'
      AND r.pollution = '{pollution}'
      AND m.run_id LIKE '{job_id}_%'

    ORDER BY m.f1_score ASC;
    """

    raw_metrics_df = run_sql_query(query)
    return raw_metrics_df


def analyze_main(jobid, dataset, lm):
    df = query_metrics(jobid)

    latex_code = generate_comparison_latex(df, test_metric='test')

    print(latex_code)

    plot_degradation_barplot(df, dataset, "main", lm)

    return df


def analyze_prelim(jobid, dataset, lm):
    df = query_prelimary(jobid)

    latex_code = generate_comparison_latex_prelim(df, dataset)

    print(latex_code)

    plot_degradation_barplot(df, dataset, "prelim", lm)
