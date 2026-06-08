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


def query_metrics_by_config(array_job, pollution, entity_type):
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
    FROM metrics m
    JOIN runs r ON m.run_id = r.run_id
    WHERE m.is_final = 1  AND m.run_id LIKE '{array_job}_%' AND r.pollution = '{pollution}' AND m.entity = '{entity_type}'
    ORDER BY r.dataset, m.entity, m.metric_type;"""

    df = run_sql_query(query)
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


def plot_retem_vs_base(jobid, filename):
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
    # sns.set_theme(style="whitegrid")
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#F1F5F9",
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}",
            # "font.family": "sans-serif",
        },
    )

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

    g = sns.JointGrid(
        data=df,
        x=X_AXIS_METRIC,
        y="f1_diff",
        height=7
    )

    for level in logical_order:
        sub_df = df[df['pollution'] == level]
        if not sub_df.empty:
            g.ax_joint.scatter(
                sub_df[X_AXIS_METRIC],
                sub_df['f1_diff'],
                label=level,
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
    x_label_text = 'RETEM F1 Score (Absolute)' if X_AXIS_METRIC == 'retem_f1' else 'Baseline F1 score'
    g.set_axis_labels(
        x_label_text, '$\Delta$ F1 score (RETEM $-$ Baseline)', fontsize=11, labelpad=10)
    # g.fig.suptitle(
    #     f"Difference F1 Score: Baseline vs RETEM - {title}", y=1.03, fontweight='bold', fontsize=13)

    # 5. Generate the crisp, discrete legend
    leg = g.ax_joint.legend(
        title="Pollution level",
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

    plt.tight_layout()

    # 4. Save/Show
    out_path = Path(f"../plots/scatter/")
    out_path.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_path / f"f1_score_{entity}.png")
    # Change this in your script:
    plt.savefig(
        out_path / f"{filename}.pdf",
        bbox_inches='tight',
        backend='pdf'
    )
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


def plot_degradation_barplot(df, dataset, rtype, lm="roberta"):
    pollution_order = ['source', 'low', 'medium', 'high']
    df['pollution'] = pd.Categorical(
        df['pollution'], categories=pollution_order, ordered=True)

    # cast to string
    df['metric_type'] = df['metric_type'].astype(str)
    # unique metric types: baseline, phase1, phase2, test
    metric_types = df['metric_type'].unique()

    entities = df['entity'].unique()

    thesis_palette = {
        "baseline": "#1E3A8A",
        "phase1": "#3B82F6",
        "phase2": "#C2410C",
        "test": "#C2410C",
    }

    for entity in entities:
        entity_df = df[df['entity'] == entity].sort_values("pollution")

        valid_data = entity_df.dropna(subset=['mean_f1'])
        if valid_data.empty:
            continue

        min_score = (entity_df['mean_f1'] -
                     entity_df['std_f1'].fillna(0)).min()
        lower_limit = min_score - 0.05

        plt.figure(figsize=(8, 5))
        # sns.set_theme(style="whitegrid")

        sns.set_theme(
            style="whitegrid",
            rc={
                "axes.edgecolor": "#CBD5E1",
                "grid.color": "#F1F5F9",
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{lmodern}",
                # "font.family": "sans-serif",
            },
        )

        ax = sns.barplot(
            data=entity_df,
            x="pollution",
            y="mean_f1",
            hue="metric_type",
            # palette="tab10"
            palette=thesis_palette
        )

        for i, metric_label in enumerate(metric_types):
            if i >= len(ax.containers):
                break

            container = ax.containers[i]

            # subset = entity_df[entity_df['metric_type']
            #                    == metric_label].sort_values("pollution")

            subset = (
                entity_df[entity_df['metric_type'] == metric_label]
                .set_index("pollution")
                .reindex(pollution_order)
                .reset_index()
            )

            valid_err_x = []
            valid_err_means = []
            valid_err_lower = []
            valid_err_upper = []

            for bar, val, std in zip(container, subset['mean_f1'].values, subset['std_f1'].values):
                if pd.isna(val):
                    continue  # Skip bars that have no data

                # Safely default zero/NaN standard deviations to 0.0
                current_std = 0.0 if pd.isna(std) or std <= 0 else std
                x_pos = bar.get_x() + bar.get_width() / 2

                # Calculate label placement apex
                if (val + current_std) > 1.0:
                    top_of_error = 1.0
                else:
                    top_of_error = val + current_std

                y_pos = top_of_error + 0.001

                # Always place the text label (even when std is 0)
                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black'
                )

                # Only collect error coordinates if a physical error band should exist
                if current_std > 0:
                    valid_err_x.append(x_pos)
                    valid_err_means.append(val)
                    valid_err_lower.append(current_std)
                    if (val + current_std) > 1.0:
                        valid_err_upper.append(1.0 - val)
                    else:
                        valid_err_upper.append(current_std)

            # Only plot error bars if valid non-zero error configurations exist
            if valid_err_x:
                ax.errorbar(
                    x=valid_err_x,
                    y=valid_err_means,
                    yerr=[valid_err_lower, valid_err_upper],
                    fmt='none',
                    c='black',
                    capsize=4,
                    elinewidth=1.2,
                    alpha=0.8
                )

            # if subset.empty:
            #     continue

            # # for bar, val in zip(container, subset['mean_f1'].values):
            # #     # Calculate X (center of the bar)
            # #     x_pos = bar.get_x() + bar.get_width() / 2
            # #     y_pos = bar.get_y()

            # #     # Position at the base: lower_limit + a tiny buffer
            # #     # va='bottom' ensures the text sits ON TOP of this coordinate
            # #     ax.text(
            # #         x=x_pos,
            # #         y=y_pos,
            # #         s=f'{val:.3f}',
            # #         ha='center',
            # #         va='bottom',
            # #         fontsize=9,
            # #         # fontweight='bold',
            # #         color='black'   # Use white so it's readable against the bar color
            # #         # rotation=90      # Vertical labels look cleaner at the bottom
            # #     )

            # x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]

            # y_stds = subset['std_f1'].values
            # # print(subset['std_f1'].values)
            # y_means = subset['mean_f1'].values

            # lower_err = y_stds
            # upper_err = []

            # for m, s in zip(y_means, y_stds):
            #     if (m + s) > 1.0:
            #         upper_err.append(1.0 - m)  # Clamp to the ceiling of 1.0
            #     else:
            #         upper_err.append(s)

            # asymmetric_err = [lower_err, upper_err]
            # # Move your error list zipper logic slightly higher or calculate upper cap on the fly:
            # for idx, (bar, val, std) in enumerate(zip(container, subset['mean_f1'].values, subset['std_f1'].values)):
            #     x_pos = bar.get_x() + bar.get_width() / 2

            #     # Calculate where the top of the error bar actually ends
            #     if (val + std) > 1.0:
            #         top_of_error = 1.0
            #     else:
            #         top_of_error = val + std

            #     # Position text slightly above the error bar tip
            #     y_pos = top_of_error + 0.001

            #     ax.text(
            #         x=x_pos,
            #         y=y_pos,
            #         s=f'{val:.3f}',
            #         ha='center',
            #         va='bottom',
            #         fontsize=9,
            #         color='black'
            #     )

            # # Only plot if we have matching data (prevents size mismatch errors)
            # if len(x_coords) == len(y_stds):
            #     ax.errorbar(
            #         x=x_coords,
            #         y=y_means,
            #         yerr=asymmetric_err,
            #         fmt='none',  # 'none' means don't connect with a line
            #         c='black',
            #         capsize=4,   # Width of the horizontal "caps"
            #         elinewidth=1.2,
            #         alpha=0.8
            #     )

        # 3. Final formatting
        plt.ylim(lower_limit, 1.03)
        # plt.title(f"Entity: {entity}")
        plt.xlabel("Pollution level")
        plt.ylabel("Mean F1 score")

        # ---------------------------------------------------------
        # CUSTOM LEGEND SIGNATURES
        # ---------------------------------------------------------
        # 1. Extract the internal handles (the colored bars) and current labels
        handles, labels = ax.get_legend_handles_labels()

        # 2. Define your clean, publication-ready display names
        legend_mapping = {
            "baseline": "Baseline",
            "phase1": "Placeholder",
            "phase2": "Oracle",
            "test": "RETEM",
        }

        # 3. Map the old labels to the new ones (falls back to original if key missing)
        new_labels = [legend_mapping.get(label, label) for label in labels]

        # 4. Re-draw the legend using the updated text strings
        ax.legend(
            handles=handles,
            labels=new_labels,
            # title="Model Variant",  # Optional: Customize the legend header text
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
        )

        # Move legend outside if it's crowded
        # plt.legend(title="Metric type", loc='upper left',
        #            bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # 4. Save/Show
        out_path = Path(f"../img/barplots/{lm}/{dataset}/{rtype}")
        out_path.mkdir(parents=True, exist_ok=True)
        # plt.savefig(out_path / f"f1_score_{entity}.png")
        # Change this in your script:
        plt.savefig(
            out_path / f"{entity}.pdf",
            bbox_inches='tight',
            backend='pdf'
        )
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
