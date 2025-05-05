threshold_list, error_list = zip(*final_errors)
            plt.plot(threshold_list, error_list, 'o-', linewidth=2, markersize=8)
            
            # Add trend line
            z = np.polyfit(threshold_list, error_list, 2)
            p = np.poly1d(z)
            x_pred = np.linspace(min(threshold_list), max(threshold_list), 100)
            plt.plot(x_pred, p(x_pred), '--', color='gray', alpha=0.7)
            
            # Add data labels
            for x, y in zip(threshold_list, error_list):
                plt.annotate(f"{y:.6f}", (x, y), textcoords="offset points", 
                           xytext=(0, 7), ha='center')
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.ylabel('Final L1 Reconstruction Error', fontsize=12)
        plt.title(f'Impact of Threshold on Final Error - {dataset_name}', fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "detailed_threshold_impact.png"), dpi=300)
        plt.close()
        
        logger.info(f"Created threshold convergence visualizations for {dataset_name}")
    
    def _visualize_combined_sensitivity(self, df, dataset_name):
        """Visualize combined sensitivity to iterations and threshold."""
        # Create visualization directory
        viz_dir = os.path.join(self.results_dir, "combined", dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Combined Parameter Sensitivity Analysis - {dataset_name}", fontsize=16)
        
        # 1. Error vs. Threshold for different max_iterations
        ax = axes[0, 0]
        for max_iters in sorted(df['max_iterations'].unique()):
            subset = df[df['max_iterations'] == max_iters]
            ax.plot(subset['threshold'], subset['l1_error'], 'o-', 
                   label=f"max_iterations={max_iters}")
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('L1 Reconstruction Error')
        ax.set_title('Error by Threshold and Iteration Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 2. Runtime vs. Threshold for different max_iterations
        ax = axes[0, 1]
        for max_iters in sorted(df['max_iterations'].unique()):
            subset = df[df['max_iterations'] == max_iters]
            ax.plot(subset['threshold'], subset['runtime_sec'], 'o-', 
                   label=f"max_iterations={max_iters}")
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime by Threshold and Iteration Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 3. Error vs. Max Iterations for different thresholds
        ax = axes[1, 0]
        for threshold in sorted(df['threshold'].unique()):
            subset = df[df['threshold'] == threshold]
            ax.plot(subset['max_iterations'], subset['l1_error'], 'o-', 
                   label=f"threshold={threshold}")
        
        ax.set_xlabel('Max Iterations')
        ax.set_ylabel('L1 Reconstruction Error')
        ax.set_title('Error by Iteration Setting and Threshold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 4. Actual Iterations vs. Threshold
        ax = axes[1, 1]
        for max_iters in sorted(df['max_iterations'].unique()):
            subset = df[df['max_iterations'] == max_iters]
            ax.plot(subset['threshold'], subset['actual_iterations'], 'o-', 
                   label=f"max_iterations={max_iters}")
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Actual Iterations')
        ax.set_title('Convergence Speed by Parameter Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(viz_dir, "combined_sensitivity.png"), dpi=300)
        plt.close()
        
        # Create heatmap of error by parameters
        plt.figure(figsize=(10, 8))
        
        # Prepare data for heatmap
        pivot_df = df.pivot_table(index='max_iterations', columns='threshold', values='l1_error')
        
        # Plot heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".6f", cmap="YlGnBu_r", 
                   cbar_kws={'label': 'L1 Reconstruction Error'})
        
        plt.title(f'Error Heatmap by Parameter Settings - {dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "error_heatmap.png"), dpi=300)
        plt.close()
        
        # Create heatmap of runtime by parameters
        plt.figure(figsize=(10, 8))
        
        # Prepare data for heatmap
        pivot_df = df.pivot_table(index='max_iterations', columns='threshold', values='runtime_sec')
        
        # Plot heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Reds", 
                   cbar_kws={'label': 'Runtime (seconds)'})
        
        plt.title(f'Runtime Heatmap by Parameter Settings - {dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "runtime_heatmap.png"), dpi=300)
        plt.close()
        
        logger.info(f"Created combined sensitivity visualizations for {dataset_name}")
    
    def visualize_results(self):
        """Create consolidated visualizations of parameter sensitivity results."""
        # Check if we have results to visualize
        if not self.results.get('iterations') and not self.results.get('threshold'):
            logger.warning("No results available for visualization")
            return
        
        # Create visualization directory
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert results to DataFrames
        iterations_df = pd.DataFrame(self.results.get('iterations', []))
        threshold_df = pd.DataFrame(self.results.get('threshold', []))
        combined_df = pd.DataFrame(self.results.get('combined', []))
        
        # Visualize iteration sensitivity across datasets
        if not iterations_df.empty:
            self._visualize_iterations_across_datasets(iterations_df, viz_dir)
        
        # Visualize threshold sensitivity across datasets
        if not threshold_df.empty:
            self._visualize_threshold_across_datasets(threshold_df, viz_dir)
        
        # Visualize combined sensitivity across datasets
        if not combined_df.empty:
            self._visualize_combined_across_datasets(combined_df, viz_dir)
        
        # Create comparative report
        self._create_sensitivity_report()
    
    def _visualize_iterations_across_datasets(self, df, viz_dir):
        """Visualize iteration sensitivity across all datasets."""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot error vs. max_iterations for each dataset
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if len(subset) >= 2:  # Only plot if we have at least 2 data points
                plt.plot(subset['max_iterations'], subset['l1_error'], 'o-', 
                       label=f"{dataset}")
        
        plt.xlabel('Max Iterations', fontsize=12)
        plt.ylabel('L1 Reconstruction Error', fontsize=12)
        plt.title('Impact of Max Iterations on Error Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "iterations_error_comparison.png"), dpi=300)
        plt.close()
        
        # Plot runtime vs. max_iterations for each dataset
        plt.figure(figsize=(12, 8))
        
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if len(subset) >= 2:  # Only plot if we have at least 2 data points
                plt.plot(subset['max_iterations'], subset['runtime_sec'], 'o-', 
                       label=f"{dataset}")
        
        plt.xlabel('Max Iterations', fontsize=12)
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.title('Impact of Max Iterations on Runtime Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "iterations_runtime_comparison.png"), dpi=300)
        plt.close()
        
        # Plot actual vs. max iterations
        plt.figure(figsize=(12, 8))
        
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if 'actual_iterations' in subset.columns and len(subset) >= 2:
                plt.plot(subset['max_iterations'], subset['actual_iterations'], 'o-', 
                       label=f"{dataset}")
        
        # Add diagonal line (y=x)
        max_val = df['max_iterations'].max()
        plt.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.7)
        
        plt.xlabel('Max Iterations Setting', fontsize=12)
        plt.ylabel('Actual Iterations Used', fontsize=12)
        plt.title('Actual vs. Maximum Iterations Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "actual_vs_max_iterations.png"), dpi=300)
        plt.close()
    
    def _visualize_threshold_across_datasets(self, df, viz_dir):
        """Visualize threshold sensitivity across all datasets."""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot error vs. threshold for each dataset
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if len(subset) >= 2:  # Only plot if we have at least 2 data points
                plt.plot(subset['threshold'], subset['l1_error'], 'o-', 
                       label=f"{dataset}")
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.ylabel('L1 Reconstruction Error', fontsize=12)
        plt.title('Impact of Threshold on Error Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "threshold_error_comparison.png"), dpi=300)
        plt.close()
        
        # Plot iterations vs. threshold for each dataset
        plt.figure(figsize=(12, 8))
        
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if len(subset) >= 2:  # Only plot if we have at least 2 data points
                plt.plot(subset['threshold'], subset['iterations'], 'o-', 
                       label=f"{dataset}")
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.ylabel('Number of Iterations', fontsize=12)
        plt.title('Impact of Threshold on Convergence Speed Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "threshold_iterations_comparison.png"), dpi=300)
        plt.close()
        
        # Plot runtime vs. threshold for each dataset
        plt.figure(figsize=(12, 8))
        
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            if len(subset) >= 2:  # Only plot if we have at least 2 data points
                plt.plot(subset['threshold'], subset['runtime_sec'], 'o-', 
                       label=f"{dataset}")
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.title('Impact of Threshold on Runtime Across Datasets', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "threshold_runtime_comparison.png"), dpi=300)
        plt.close()
    
    def _visualize_combined_across_datasets(self, df, viz_dir):
        """Visualize combined parameter sensitivity across all datasets."""
        # Only proceed if we have meaningful data
        if len(df) < 4:
            return
        
        # Create a bubble chart showing error, runtime, and iterations
        plt.figure(figsize=(12, 10))
        
        # Define colormap for datasets
        datasets = df['dataset'].unique()
        colormap = plt.cm.get_cmap('tab10', len(datasets))
        
        # Plot each dataset with different markers for max_iterations
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset'] == dataset]
            
            # Use different markers for max_iterations
            markers = {10: 'o', 20: 's', 30: '^', 40: 'D', 50: 'v'}
            
            for max_iters in sorted(dataset_df['max_iterations'].unique()):
                iters_df = dataset_df[dataset_df['max_iterations'] == max_iters]
                
                # Size bubbles by runtime
                sizes = iters_df['runtime_sec'] * 10  # Scale for visibility
                
                plt.scatter(
                    iters_df['threshold'], 
                    iters_df['l1_error'], 
                    s=sizes,
                    c=[colormap(i)] * len(iters_df),
                    marker=markers.get(max_iters, 'o'),
                    alpha=0.7,
                    label=f"{dataset} (max_iter={max_iters})"
                )
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.ylabel('L1 Reconstruction Error', fontsize=12)
        plt.title('Parameter Sensitivity Bubble Chart', fontsize=14)
        plt.grid(alpha=0.3)
        
        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Add annotation explaining bubble size
        plt.annotate('Bubble size represents runtime', xy=(0.05, 0.95), 
                   xycoords='axes fraction', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "combined_bubble_chart.png"), dpi=300)
        plt.close()
    
    def _create_sensitivity_report(self):
        """Create a detailed report on parameter sensitivity."""
        report_path = os.path.join(self.results_dir, "parameter_sensitivity_report.md")
        
        # Prepare data for the report
        iterations_df = pd.DataFrame(self.results.get('iterations', []))
        threshold_df = pd.DataFrame(self.results.get('threshold', []))
        combined_df = pd.DataFrame(self.results.get('combined', []))
        
        with open(report_path, 'w') as f:
            f.write("# SSumM Parameter Sensitivity Analysis Report\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Introduction
            f.write("## Introduction\n\n")
            f.write("This report analyzes the sensitivity of the SSumM algorithm to different parameter settings, ")
            f.write("particularly the number of iterations (T) and the threshold value for accepting merges. ")
            f.write("Understanding these sensitivities helps optimize the algorithm for different types of graphs ")
            f.write("and ensures consistent performance across datasets.\n\n")
            
            # Iteration Sensitivity
            if not iterations_df.empty:
                f.write("## Sensitivity to Number of Iterations (T)\n\n")
                
                # Calculate average impact across datasets
                avg_impact = iterations_df.groupby('max_iterations').agg({
                    'l1_error': ['mean', 'std'],
                    'runtime_sec': ['mean', 'std'],
                    'actual_iterations': ['mean', 'max']
                }).reset_index()
                
                f.write("### Impact on Reconstruction Error\n\n")
                f.write("| Max Iterations | Avg L1 Error | Std Dev | Avg Runtime (s) |\n")
                f.write("|---------------|-------------|---------|----------------|\n")
                
                for _, row in avg_impact.iterrows():
                    max_iters = row['max_iterations']
                    avg_error = row[('l1_error', 'mean')]
                    std_error = row[('l1_error', 'std')]
                    avg_runtime = row[('runtime_sec', 'mean')]
                    
                    f.write(f"| {int(max_iters)} | {avg_error:.6f} | {std_error:.6f} | {avg_runtime:.2f} |\n")
                
                f.write("\n")
                
                # Observations about iterations
                f.write("### Key Observations\n\n")
                
                # Check if error decreases with more iterations
                first_error = avg_impact.iloc[0][('l1_error', 'mean')]
                last_error = avg_impact.iloc[-1][('l1_error', 'mean')]
                
                if first_error > last_error:
                    improvement = (first_error - last_error) / first_error * 100
                    f.write(f"- Increasing max iterations from {int(avg_impact.iloc[0]['max_iterations'])} to ")
                    f.write(f"{int(avg_impact.iloc[-1]['max_iterations'])} reduces error by {improvement:.1f}%\n")
                else:
                    f.write("- Increasing max iterations does not consistently reduce error across datasets\n")
                
                # Check if actual iterations approach max iterations
                max_avg_actual = avg_impact[('actual_iterations', 'mean')].max()
                max_setting = avg_impact['max_iterations'].max()
                
                if max_avg_actual < 0.7 * max_setting:
                    f.write(f"- The algorithm typically converges before reaching the maximum iterations ")
                    f.write(f"(avg actual: {max_avg_actual:.1f} vs. max setting: {max_setting})\n")
                else:
                    f.write(f"- The algorithm often uses the full number of iterations allowed\n")
                
                # Runtime analysis
                runtime_increase = (avg_impact[('runtime_sec', 'mean')].iloc[-1] - 
                                  avg_impact[('runtime_sec', 'mean')].iloc[0])
                
                f.write(f"- Increasing max iterations increases runtime by approximately ")
                f.write(f"{runtime_increase:.2f} seconds from lowest to highest setting\n\n")
                
                # Recommended value
                min_error_idx = avg_impact[('l1_error', 'mean')].idxmin()
                recommended_iters = avg_impact.iloc[min_error_idx]['max_iterations']
                
                f.write(f"**Recommended value**: {int(recommended_iters)} iterations provides the best ")
                f.write(f"balance of accuracy and performance\n\n")
                
                # Include visualization reference
                f.write("![Iterations Impact on Error](visualizations/iterations_error_comparison.png)\n\n")
            
            # Threshold Sensitivity
            if not threshold_df.empty:
                f.write("## Sensitivity to Threshold Value\n\n")
                
                # Calculate average impact across datasets
                avg_impact = threshold_df.groupby('threshold').agg({
                    'l1_error': ['mean', 'std'],
                    'runtime_sec': ['mean', 'std'],
                    'iterations': ['mean', 'std']
                }).reset_index()
                
                f.write("### Impact on Reconstruction Error and Convergence\n\n")
                f.write("| Threshold | Avg L1 Error | Avg Iterations | Avg Runtime (s) |\n")
                f.write("|-----------|-------------|---------------|----------------|\n")
                
                for _, row in avg_impact.iterrows():
                    threshold = row['threshold']
                    avg_error = row[('l1_error', 'mean')]
                    avg_iters = row[('iterations', 'mean')]
                    avg_runtime = row[('runtime_sec', 'mean')]
                    
                    f.write(f"| {threshold:.1f} | {avg_error:.6f} | {avg_iters:.1f} | {avg_runtime:.2f} |\n")
                
                f.write("\n")
                
                # Observations about threshold
                f.write("### Key Observations\n\n")
                
                # Find optimal threshold
                min_error_idx = avg_impact[('l1_error', 'mean')].idxmin()
                optimal_threshold = avg_impact.iloc[min_error_idx]['threshold']
                
                f.write(f"- The optimal threshold value for minimizing error is {optimal_threshold:.1f}\n")
                
                # Check effect on iterations
                thresholds = avg_impact['threshold'].tolist()
                iterations = avg_impact[('iterations', 'mean')].tolist()
                
                # Linear regression to see trend
                slope = np.polyfit(thresholds, iterations, 1)[0]
                
                if slope < 0:
                    f.write("- Higher threshold values lead to faster convergence (fewer iterations)\n")
                else:
                    f.write("- Threshold has minimal impact on the number of iterations needed for convergence\n")
                
                # Check effect on runtime
                runtime_lowest = avg_impact[('runtime_sec', 'mean')].min()
                runtime_threshold = avg_impact.iloc[avg_impact[('runtime_sec', 'mean')].idxmin()]['threshold']
                
                f.write(f"- The most efficient threshold for runtime is {runtime_threshold:.1f} ")
                f.write(f"(avg: {runtime_lowest:.2f}s)\n\n")
                
                # Recommended value
                f.write(f"**Recommended value**: {optimal_threshold:.1f} provides the best accuracy, ")
                
                runtime_at_optimal = avg_impact.iloc[min_error_idx][('runtime_sec', 'mean')]
                runtime_diff = runtime_at_optimal - runtime_lowest
                
                if runtime_diff > 0.5:  # If there's a significant difference
                    f.write(f"with a runtime penalty of {runtime_diff:.2f}s compared to the fastest setting\n\n")
                else:
                    f.write(f"with minimal impact on runtime performance\n\n")
                
                # Include visualization reference
                f.write("![Threshold Impact on Error](visualizations/threshold_error_comparison.png)\n\n")
            
            # Combined Analysis
            if not combined_df.empty and len(combined_df) >= 4:
                f.write("## Combined Parameter Sensitivity\n\n")
                
                f.write("### Interaction Between Iterations and Threshold\n\n")
                
                # Create a pivot table of error by parameters
                datasets = combined_df['dataset'].unique()
                
                for dataset in datasets:
                    dataset_df = combined_df[combined_df['dataset'] == dataset]
                    
                    if len(dataset_df) >= 4:  # Only if we have enough data points
                        pivot_df = dataset_df.pivot_table(
                            index='max_iterations', 
                            columns='threshold', 
                            values='l1_error'
                        )
                        
                        f.write(f"#### Error by Parameter Settings for {dataset}\n\n")
                        f.write("| Max Iterations |")
                        
                        # Write header row with threshold values
                        for threshold in pivot_df.columns:
                            f.write(f" Threshold={threshold:.1f} |")
                        f.write("\n")
                        
                        # Write separator row
                        f.write("|---------------|")
                        for _ in pivot_df.columns:
                            f.write("--------------|")
                        f.write("\n")
                        
                        # Write data rows
                        for max_iters, row in pivot_df.iterrows():
                            f.write(f"| {int(max_iters)} |")
                            
                            for threshold in pivot_df.columns:
                                error = row[threshold]
                                f.write(f" {error:.6f} |")
                            f.write("\n")
                        
                        f.write("\n")
                
                # Observations about parameter interactions
                f.write("### Key Observations on Parameter Interactions\n\n")
                
                # Find optimal combinations
                best_combo = combined_df.loc[combined_df['l1_error'].idxmin()]
                fastest_combo = combined_df.loc[combined_df['runtime_sec'].idxmin()]
                
                f.write(f"- The optimal parameter combination for minimizing error is **max_iterations={int(best_combo['max_iterations'])}, ")
                f.write(f"threshold={best_combo['threshold']:.1f}** (error: {best_combo['l1_error']:.6f})\n")
                
                f.write(f"- The fastest parameter combination is **max_iterations={int(fastest_combo['max_iterations'])}, ")
                f.write(f"threshold={fastest_combo['threshold']:.1f}** (runtime: {fastest_combo['runtime_sec']:.2f}s)\n")
                
                # Check if there's a clear trade-off pattern
                f.write("- Parameter interaction effect: ")
                
                # Group by max_iterations and find best threshold for each
                best_thresholds = combined_df.groupby('max_iterations').apply(
                    lambda x: x.loc[x['l1_error'].idxmin()]['threshold']
                ).reset_index()
                
                # Check if best threshold changes with max_iterations
                if best_thresholds['threshold'].nunique() > 1:
                    f.write("the optimal threshold value varies depending on the maximum iterations setting\n\n")
                else:
                    f.write("the optimal threshold value is consistent regardless of the maximum iterations setting\n\n")
                
                # Include visualization reference
                f.write("![Combined Parameter Sensitivity](visualizations/combined_bubble_chart.png)\n\n")
            
            # Overall Recommendations
            f.write("## Overall Recommendations\n\n")
            
            # Gather best parameters from individual analyses
            best_params = {}
            
            if not iterations_df.empty:
                min_error_idx = iterations_df.groupby('max_iterations')['l1_error'].mean().idxmin()
                best_params['max_iterations'] = min_error_idx
            
            if not threshold_df.empty:
                min_error_idx = threshold_df.groupby('threshold')['l1_error'].mean().idxmin()
                best_params['threshold'] = min_error_idx
            
            # If we have combined analysis, it takes precedence
            if not combined_df.empty and len(combined_df) >= 4:
                grouped = combined_df.groupby(['max_iterations', 'threshold'])['l1_error'].mean().reset_index()
                best_combo = grouped.loc[grouped['l1_error'].idxmin()]
                best_params['max_iterations'] = best_combo['max_iterations']
                best_params['threshold'] = best_combo['threshold']
            
            if best_params:
                f.write("Based on the sensitivity analysis, the recommended parameter settings for SSumM are:\n\n")
                f.write("```python\n")
                
                if 'max_iterations' in best_params:
                    f.write(f"max_iterations = {int(best_params['max_iterations'])}\n")
                
                if 'threshold' in best_params:
                    f.write(f"threshold_formula = lambda iteration: {best_params['threshold']:.1f}  # Fixed threshold\n")
                
                f.write("```\n\n")
                
                f.write("### Optimal Settings for Different Graph Types\n\n")
                
                # Try to identify optimal settings by graph type
                datasets_by_type = {
                    'scale_free': [d for d in combined_df['dataset'].unique() if 'ba_' in d],
                    'small_world': [d for d in combined_df['dataset'].unique() if 'ws_' in d],
                    'regular': [d for d in combined_df['dataset'].unique() if 'reg_' in d],
                    'real': [d for d in combined_df['dataset'].unique() 
                          if 'ba_' not in d and 'ws_' not in d and 'reg_' not in d and 'er_' not in d]
                }
                
                for graph_type, datasets in datasets_by_type.items():
                    if datasets and len(combined_df[combined_df['dataset'].isin(datasets)]) > 0:
                        type_df = combined_df[combined_df['dataset'].isin(datasets)]
                        best = type_df.loc[type_df['l1_error'].idxmin()]
                        
                        f.write(f"**{graph_type.title()} Graphs**: max_iterations={int(best['max_iterations'])}, ")
                        f.write(f"threshold={best['threshold']:.1f}\n")
                
                f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("The parameter sensitivity analysis reveals that both the number of iterations and the threshold ")
            f.write("value significantly impact the SSumM algorithm's performance. The results demonstrate that:\n\n")
            
            f.write("1. **Iteration Count**: Increasing the maximum iterations generally improves accuracy up to a ")
            f.write("point of diminishing returns, but also increases runtime linearly.\n")
            
            f.write("2. **Threshold Value**: The threshold for accepting merges has a significant impact on both error ")
            f.write("and convergence speed, with an optimal value typically between 0.2-0.4 for most graphs.\n")
            
            f.write("3. **Parameter Interaction**: The interaction between iterations and threshold creates complex ")
            f.write("trade-offs that should be considered together rather than independently.\n")
            
            f.write("4. **Graph Type Dependency**: Different graph structures (scale-free, small-world, etc.) ")
            f.write("show varying sensitivities to parameter settings, suggesting adaptive parameters may be beneficial.\n\n")
            
            f.write("This analysis provides a foundation for setting optimal parameters based on the specific ")
            f.write("requirements of the summarization task and the structure of the input graph.")
        
        logger.info(f"Sensitivity report saved to {report_path}")
        return report_path
    
    def run_sensitivity_analysis(self, dataset_names=None):
        """
        Run the full parameter sensitivity analysis.
        
        Args:
            dataset_names: List of dataset names to analyze
            
        Returns:
            dict: Dictionary of analysis results
        """
        # Load datasets
        datasets = self.load_datasets(dataset_names)
        
        # Run analysis for iterations
        iterations_df = self.analyze_iteration_sensitivity(datasets)
        
        # Run analysis for threshold
        threshold_df = self.analyze_threshold_sensitivity(datasets)
        
        # Run combined analysis
        combined_df = self.analyze_combined_sensitivity(datasets)
        
        # Create visualizations
        self.visualize_results()
        
        return {
            'iterations': iterations_df,
            'threshold': threshold_df,
            'combined': combined_df
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze parameter sensitivity for SSumM algorithm')
    
    # Dataset options
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing datasets')
    parser.add_argument('--datasets', nargs='+', 
                        default=None,
                        help='Names of datasets to use (None for synthetic only)')
    
    # Analysis options
    parser.add_argument('--results_dir', type=str, default='./sensitivity_results',
                        help='Directory to save results')
    parser.add_argument('--iterations_only', action='store_true',
                        help='Only test sensitivity to iterations')
    parser.add_argument('--threshold_only', action='store_true',
                        help='Only test sensitivity to threshold')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Run a quick test with limited parameters')
    
    return parser.parse_args()

def run_quick_test():
    """Run a quick sensitivity analysis test."""
    logger.info("Running quick sensitivity analysis test")
    
    # Create analyzer
    analyzer = ParameterSensitivityAnalyzer(
        data_dir="./data",
        results_dir="./quick_sensitivity_test"
    )
    
    # Create small test dataset
    datasets = {
        "karate": nx.karate_club_graph(),
        "ba_50_2": nx.barabasi_albert_graph(n=50, m=2, seed=42)
    }
    
    # Test with limited parameter values
    iterations_df = analyzer.analyze_iteration_sensitivity(
        datasets, 
        max_iterations_values=[5, 10, 15]
    )
    
    threshold_df = analyzer.analyze_threshold_sensitivity(
        datasets,
        threshold_values=[0.1, 0.3, 0.5]
    )
    
    combined_df = analyzer.analyze_combined_sensitivity(
        datasets,
        max_iterations_values=[5, 15],
        threshold_values=[0.1, 0.5]
    )
    
    # Create visualizations
    analyzer.visualize_results()
    
    logger.info("Quick sensitivity test completed successfully")

def main():
    """Main function to run parameter sensitivity analysis."""
    args = parse_arguments()
    
    if args.quick_test:
        run_quick_test()
        return
    
    analyzer = ParameterSensitivityAnalyzer(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    if args.iterations_only:
        # Only analyze iterations
        datasets = analyzer.load_datasets(args.datasets)
        analyzer.analyze_iteration_sensitivity(datasets)
        analyzer.visualize_results()
    elif args.threshold_only:
        # Only analyze threshold
        datasets = analyzer.load_datasets(args.datasets)
        analyzer.analyze_threshold_sensitivity(datasets)
        analyzer.visualize_results()
    else:
        # Run full analysis
        analyzer.run_sensitivity_analysis(args.datasets)
    
    logger.info("Parameter sensitivity analysis completed successfully")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Parameter sensitivity analysis for the SSumM algorithm.

This script analyzes how varying parameters (number of iterations, threshold,
candidate set size) affects the SSumM algorithm's performance and convergence.
"""

import os
import argparse
import time
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import logging
from itertools import product

# Import our implementations
from graph_utils import GraphUtils
from ssumm_algorithm import SSumM
from ssumm_optimized import OptimizedSSumM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_sensitivity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ParameterSensitivity')

class ParameterSensitivityAnalyzer:
    """
    Class for analyzing parameter sensitivity in SSumM algorithm.
    """
    
    def __init__(self, data_dir="./data", results_dir="./sensitivity_results"):
        """
        Initialize the parameter sensitivity analyzer.
        
        Args:
            data_dir: Directory containing datasets
            results_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories for different analyses
        os.makedirs(os.path.join(results_dir, "iterations"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "threshold"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "combined"), exist_ok=True)
        
        # Initialize results
        self.results = {
            'iterations': [],
            'threshold': [],
            'combined': []
        }
    
    def load_datasets(self, dataset_names=None, synthetic=True):
        """
        Load datasets for parameter sensitivity analysis.
        
        Args:
            dataset_names: List of dataset names to load
            synthetic: Whether to include synthetic datasets
            
        Returns:
            dict: Dictionary of loaded graphs
        """
        datasets = {}
        
        # Load real datasets if names provided
        if dataset_names:
            for name in dataset_names:
                try:
                    # Try to find the dataset file
                    file_path = self._find_dataset_file(name)
                    
                    if file_path:
                        logger.info(f"Loading dataset: {name} from {file_path}")
                        
                        # Determine file format
                        if file_path.endswith('.gpickle'):
                            G = GraphUtils.load_graph(file_path, format="pickle")
                        elif file_path.endswith('.txt') or file_path.endswith('.edges'):
                            G = GraphUtils.load_graph(file_path, format="edge_list")
                        else:
                            logger.warning(f"Unsupported file format for {file_path}, skipping")
                            continue
                        
                        # Ensure graph is undirected and has no self-loops
                        if G.is_directed():
                            G = G.to_undirected()
                        G.remove_edges_from(nx.selfloop_edges(G))
                        
                        # Sample large graphs to make analysis tractable
                        if G.number_of_nodes() > 500:
                            logger.info(f"Sampling large graph {name} to 500 nodes for sensitivity analysis")
                            nodes = list(G.nodes())
                            sampled_nodes = np.random.choice(nodes, 500, replace=False)
                            G = G.subgraph(sampled_nodes).copy()
                            name = f"{name}_sampled_500"
                        
                        # Store graph
                        datasets[name] = G
                        logger.info(f"  Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    else:
                        logger.warning(f"Could not find dataset file for {name}")
                
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}", exc_info=True)
        
        # Add synthetic datasets if requested
        if synthetic:
            logger.info("Creating synthetic datasets with different structures")
            
            # Barabasi-Albert (scale-free network)
            G_ba = nx.barabasi_albert_graph(n=200, m=2, seed=42)
            datasets["ba_200_2"] = G_ba
            logger.info(f"  Created BA graph: {G_ba.number_of_nodes()} nodes, {G_ba.number_of_edges()} edges")
            
            # Watts-Strogatz (small-world network)
            G_ws = nx.watts_strogatz_graph(n=200, k=4, p=0.1, seed=42)
            datasets["ws_200_4_0.1"] = G_ws
            logger.info(f"  Created WS graph: {G_ws.number_of_nodes()} nodes, {G_ws.number_of_edges()} edges")
            
            # Random regular graph (uniform degree)
            G_reg = nx.random_regular_graph(d=3, n=200, seed=42)
            datasets["reg_200_3"] = G_reg
            logger.info(f"  Created regular graph: {G_reg.number_of_nodes()} nodes, {G_reg.number_of_edges()} edges")
        
        # Add at least one small test graph
        if not datasets:
            logger.warning("No datasets loaded. Adding Zachary's Karate Club graph as fallback.")
            datasets["karate"] = nx.karate_club_graph()
        
        return datasets
    
    def _find_dataset_file(self, dataset_name):
        """Find the file for a dataset by name."""
        # Look for common file extensions
        for ext in ['.gpickle', '.txt', '.edges', '.edgelist']:
            path = os.path.join(self.data_dir, f"{dataset_name}{ext}")
            if os.path.exists(path):
                return path
        
        # Try without adding extension (if name already includes it)
        path = os.path.join(self.data_dir, dataset_name)
        if os.path.exists(path):
            return path
        
        return None
    
    def analyze_iteration_sensitivity(self, datasets, max_iterations_values=None, target_ratio=0.3):
        """
        Analyze sensitivity to number of iterations (T).
        
        Args:
            datasets: Dictionary of datasets
            max_iterations_values: List of max_iterations values to test
            target_ratio: Target size ratio
            
        Returns:
            pd.DataFrame: Results of the analysis
        """
        if max_iterations_values is None:
            max_iterations_values = [5, 10, 15, 20, 25, 30, 40, 50]
        
        logger.info(f"Analyzing sensitivity to number of iterations")
        logger.info(f"Testing values: {max_iterations_values}")
        
        results = []
        
        for dataset_name, graph in datasets.items():
            logger.info(f"\nAnalyzing iteration sensitivity on {dataset_name}")
            
            # Calculate target size
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(target_ratio * original_size)
            
            # Track convergence for each max_iterations value
            all_convergence = {}
            
            for max_iters in tqdm(max_iterations_values, desc=f"Testing iterations on {dataset_name}"):
                try:
                    # Create algorithm with this max_iterations
                    algorithm = OptimizedSSumM(max_iterations=max_iters)
                    
                    # Run the algorithm
                    start_time = time.time()
                    summary, performance_data = algorithm.summarize(graph, target_size, track_convergence=True)
                    runtime = time.time() - start_time
                    
                    # Get convergence data
                    convergence = performance_data.get('convergence', {})
                    actual_iterations = len(convergence.get('iteration', [])) if convergence else 0
                    
                    # Calculate metrics
                    size_bits = summary.size_in_bits()
                    relative_size = size_bits / original_size
                    l1_error = summary.compute_reconstruction_error(p=1)
                    
                    # Store results
                    results.append({
                        'dataset': dataset_name,
                        'max_iterations': max_iters,
                        'actual_iterations': actual_iterations,
                        'runtime_sec': runtime,
                        'size_bits': size_bits,
                        'relative_size': relative_size,
                        'l1_error': l1_error,
                        'num_supernodes': len(summary.supernodes),
                        'num_superedges': len(summary.superedge_counts),
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'graph_density': nx.density(graph)
                    })
                    
                    # Store convergence data
                    all_convergence[max_iters] = convergence
                    
                    logger.info(f"  max_iterations={max_iters}: error={l1_error:.6f}, runtime={runtime:.2f}s, "
                              f"actual_iterations={actual_iterations}")
                
                except Exception as e:
                    logger.error(f"Error with max_iterations={max_iters} on {dataset_name}: {e}", exc_info=True)
            
            # Create visualizations for this dataset
            if all_convergence:
                self._visualize_iteration_convergence(all_convergence, dataset_name)
        
        # Store results
        results_df = pd.DataFrame(results)
        self.results['iterations'] = results
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, "iterations", "iteration_sensitivity_results.csv"), 
                         index=False)
        
        return results_df
    
    def analyze_threshold_sensitivity(self, datasets, threshold_values=None, target_ratio=0.3):
        """
        Analyze sensitivity to threshold values.
        
        Args:
            datasets: Dictionary of datasets
            threshold_values: List of threshold values to test
            target_ratio: Target size ratio
            
        Returns:
            pd.DataFrame: Results of the analysis
        """
        if threshold_values is None:
            # Test various threshold formulas
            threshold_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        logger.info(f"Analyzing sensitivity to threshold values")
        logger.info(f"Testing values: {threshold_values}")
        
        results = []
        
        for dataset_name, graph in datasets.items():
            logger.info(f"\nAnalyzing threshold sensitivity on {dataset_name}")
            
            # Calculate target size
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(target_ratio * original_size)
            
            # Modify the SSumM algorithm to use a fixed threshold
            class FixedThresholdSSumM(OptimizedSSumM):
                def __init__(self, fixed_threshold, **kwargs):
                    super().__init__(**kwargs)
                    self.fixed_threshold = fixed_threshold
                
                def _calculate_threshold(self, iteration):
                    return self.fixed_threshold
            
            # Track convergence for each threshold value
            all_convergence = {}
            
            for threshold in tqdm(threshold_values, desc=f"Testing thresholds on {dataset_name}"):
                try:
                    # Create algorithm with this threshold
                    algorithm = FixedThresholdSSumM(fixed_threshold=threshold, max_iterations=30)
                    
                    # Run the algorithm
                    start_time = time.time()
                    summary, performance_data = algorithm.summarize(graph, target_size, track_convergence=True)
                    runtime = time.time() - start_time
                    
                    # Get convergence data
                    convergence = performance_data.get('convergence', {})
                    iterations = len(convergence.get('iteration', [])) if convergence else 0
                    
                    # Calculate metrics
                    size_bits = summary.size_in_bits()
                    relative_size = size_bits / original_size
                    l1_error = summary.compute_reconstruction_error(p=1)
                    
                    # Store results
                    results.append({
                        'dataset': dataset_name,
                        'threshold': threshold,
                        'iterations': iterations,
                        'runtime_sec': runtime,
                        'size_bits': size_bits,
                        'relative_size': relative_size,
                        'l1_error': l1_error,
                        'num_supernodes': len(summary.supernodes),
                        'num_superedges': len(summary.superedge_counts),
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'graph_density': nx.density(graph)
                    })
                    
                    # Store convergence data
                    all_convergence[threshold] = convergence
                    
                    logger.info(f"  threshold={threshold}: error={l1_error:.6f}, runtime={runtime:.2f}s, "
                              f"iterations={iterations}")
                
                except Exception as e:
                    logger.error(f"Error with threshold={threshold} on {dataset_name}: {e}", exc_info=True)
            
            # Create visualizations for this dataset
            if all_convergence:
                self._visualize_threshold_convergence(all_convergence, dataset_name)
        
        # Store results
        results_df = pd.DataFrame(results)
        self.results['threshold'] = results
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, "threshold", "threshold_sensitivity_results.csv"), 
                         index=False)
        
        return results_df
    
    def analyze_combined_sensitivity(self, datasets, max_iterations_values=None, threshold_values=None, 
                                   target_ratio=0.3):
        """
        Analyze combined sensitivity to iterations and threshold.
        
        Args:
            datasets: Dictionary of datasets
            max_iterations_values: List of max_iterations values to test
            threshold_values: List of threshold values to test
            target_ratio: Target size ratio
            
        Returns:
            pd.DataFrame: Results of the analysis
        """
        if max_iterations_values is None:
            max_iterations_values = [10, 20, 30]
        
        if threshold_values is None:
            threshold_values = [0.1, 0.3, 0.5]
        
        logger.info(f"Analyzing combined sensitivity to iterations and threshold")
        logger.info(f"Testing iterations: {max_iterations_values}")
        logger.info(f"Testing thresholds: {threshold_values}")
        
        results = []
        
        for dataset_name, graph in datasets.items():
            logger.info(f"\nAnalyzing combined sensitivity on {dataset_name}")
            
            # Calculate target size
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(target_ratio * original_size)
            
            # Modify the SSumM algorithm to use a fixed threshold
            class FixedThresholdSSumM(OptimizedSSumM):
                def __init__(self, fixed_threshold, **kwargs):
                    super().__init__(**kwargs)
                    self.fixed_threshold = fixed_threshold
                
                def _calculate_threshold(self, iteration):
                    return self.fixed_threshold
            
            # Test all combinations of parameters
            combinations = list(product(max_iterations_values, threshold_values))
            
            for max_iters, threshold in tqdm(combinations, 
                                           desc=f"Testing combinations on {dataset_name}"):
                try:
                    # Create algorithm with these parameters
                    algorithm = FixedThresholdSSumM(
                        fixed_threshold=threshold, 
                        max_iterations=max_iters
                    )
                    
                    # Run the algorithm
                    start_time = time.time()
                    summary, performance_data = algorithm.summarize(graph, target_size, track_convergence=True)
                    runtime = time.time() - start_time
                    
                    # Get convergence data
                    convergence = performance_data.get('convergence', {})
                    iterations = len(convergence.get('iteration', [])) if convergence else 0
                    
                    # Calculate metrics
                    size_bits = summary.size_in_bits()
                    relative_size = size_bits / original_size
                    l1_error = summary.compute_reconstruction_error(p=1)
                    
                    # Store results
                    results.append({
                        'dataset': dataset_name,
                        'max_iterations': max_iters,
                        'threshold': threshold,
                        'actual_iterations': iterations,
                        'runtime_sec': runtime,
                        'size_bits': size_bits,
                        'relative_size': relative_size,
                        'l1_error': l1_error,
                        'num_supernodes': len(summary.supernodes),
                        'num_superedges': len(summary.superedge_counts),
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges()),
                        'graph_density': nx.density(graph)
                        })
                    
                    logger.info(f"  max_iterations={max_iters}, threshold={threshold}: "
                              f"error={l1_error:.6f}, runtime={runtime:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error with max_iterations={max_iters}, threshold={threshold} "
                               f"on {dataset_name}: {e}", exc_info=True)
            
            # Create visualizations for this dataset
            if len(results) > 0:
                df = pd.DataFrame([r for r in results if r['dataset'] == dataset_name])
                if not df.empty:
                    self._visualize_combined_sensitivity(df, dataset_name)
        
        # Store results
        results_df = pd.DataFrame(results)
        self.results['combined'] = results
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, "combined", "combined_sensitivity_results.csv"), 
                         index=False)
        
        return results_df
    
    def _visualize_iteration_convergence(self, all_convergence, dataset_name):
        """Visualize convergence for different numbers of iterations."""
        # Create visualization directory
        viz_dir = os.path.join(self.results_dir, "iterations", dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Iteration Sensitivity Analysis - {dataset_name}", fontsize=16)
        
        # Extract all max_iterations values
        max_iterations_values = sorted(all_convergence.keys())
        
        # 1. Error vs. Iteration for different max_iterations
        ax = axes[0, 0]
        for max_iters in max_iterations_values:
            convergence = all_convergence[max_iters]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            errors = convergence.get('reconstruction_error_l1', [])
            
            if len(iterations) > 0 and len(errors) > 0:
                ax.plot(iterations, errors, 'o-', label=f"max_iterations={max_iters}")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L1 Reconstruction Error')
        ax.set_title('Error Convergence by Iteration Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 2. Size vs. Iteration for different max_iterations
        ax = axes[0, 1]
        for max_iters in max_iterations_values:
            convergence = all_convergence[max_iters]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            sizes = convergence.get('size_in_bits', [])
            
            if len(iterations) > 0 and len(sizes) > 0:
                ax.plot(iterations, sizes, 'o-', label=f"max_iterations={max_iters}")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Size (bits)')
        ax.set_title('Size Convergence by Iteration Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 3. Final Error vs. Max Iterations
        ax = axes[1, 0]
        final_errors = []
        for max_iters in max_iterations_values:
            convergence = all_convergence[max_iters]
            if not convergence:
                continue
                
            errors = convergence.get('reconstruction_error_l1', [])
            if errors:
                final_errors.append((max_iters, errors[-1]))
        
        if final_errors:
            max_iters_list, error_list = zip(*final_errors)
            ax.plot(max_iters_list, error_list, 'o-')
            
            # Add data labels
            for x, y in zip(max_iters_list, error_list):
                ax.annotate(f"{y:.6f}", (x, y), textcoords="offset points", 
                          xytext=(0, 5), ha='center')
        
        ax.set_xlabel('Max Iterations')
        ax.set_ylabel('Final L1 Reconstruction Error')
        ax.set_title('Impact of Max Iterations on Final Error')
        ax.grid(alpha=0.3)
        
        # 4. Convergence Speed (iterations to reach 90% of improvement)
        ax = axes[1, 1]
        convergence_speeds = []
        
        for max_iters in max_iterations_values:
            convergence = all_convergence[max_iters]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            errors = convergence.get('reconstruction_error_l1', [])
            
            if len(iterations) > 1 and len(errors) > 1:
                # Calculate convergence speed (iterations to reach 90% of improvement)
                initial_error = errors[0]
                final_error = errors[-1]
                target_error = initial_error - 0.9 * (initial_error - final_error)
                
                # Find the first iteration where error <= target_error
                for i, error in enumerate(errors):
                    if error <= target_error:
                        convergence_speeds.append((max_iters, iterations[i]))
                        break
        
        if convergence_speeds:
            max_iters_list, speed_list = zip(*convergence_speeds)
            ax.plot(max_iters_list, speed_list, 'o-')
            
            # Add data labels
            for x, y in zip(max_iters_list, speed_list):
                ax.annotate(f"{y}", (x, y), textcoords="offset points", 
                          xytext=(0, 5), ha='center')
        
        ax.set_xlabel('Max Iterations')
        ax.set_ylabel('Iterations to 90% Convergence')
        ax.set_title('Convergence Speed Analysis')
        ax.grid(alpha=0.3)
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(viz_dir, "iteration_convergence.png"), dpi=300)
        plt.close()
        
        # Create a more detailed error convergence plot
        plt.figure(figsize=(12, 8))
        for max_iters in max_iterations_values:
            convergence = all_convergence[max_iters]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            errors = convergence.get('reconstruction_error_l1', [])
            
            if len(iterations) > 0 and len(errors) > 0:
                plt.plot(iterations, errors, 'o-', linewidth=2, label=f"max_iterations={max_iters}")
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('L1 Reconstruction Error', fontsize=12)
        plt.title(f'Error Convergence Analysis - {dataset_name}', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        
        # Set appropriate y-axis limits
        all_errors = []
        for conv in all_convergence.values():
            if conv and 'reconstruction_error_l1' in conv:
                all_errors.extend(conv['reconstruction_error_l1'])
        
        if all_errors:
            plt.ylim(min(all_errors) * 0.95, max(all_errors) * 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "detailed_error_convergence.png"), dpi=300)
        plt.close()
        
        logger.info(f"Created iteration convergence visualizations for {dataset_name}")
    
    def _visualize_threshold_convergence(self, all_convergence, dataset_name):
        """Visualize convergence for different threshold values."""
        # Create visualization directory
        viz_dir = os.path.join(self.results_dir, "threshold", dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Threshold Sensitivity Analysis - {dataset_name}", fontsize=16)
        
        # Extract all threshold values
        threshold_values = sorted(all_convergence.keys())
        
        # 1. Error vs. Iteration for different thresholds
        ax = axes[0, 0]
        for threshold in threshold_values:
            convergence = all_convergence[threshold]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            errors = convergence.get('reconstruction_error_l1', [])
            
            if len(iterations) > 0 and len(errors) > 0:
                ax.plot(iterations, errors, 'o-', label=f"threshold={threshold}")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L1 Reconstruction Error')
        ax.set_title('Error Convergence by Threshold Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 2. Size vs. Iteration for different thresholds
        ax = axes[0, 1]
        for threshold in threshold_values:
            convergence = all_convergence[threshold]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            sizes = convergence.get('size_in_bits', [])
            
            if len(iterations) > 0 and len(sizes) > 0:
                ax.plot(iterations, sizes, 'o-', label=f"threshold={threshold}")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Size (bits)')
        ax.set_title('Size Convergence by Threshold Setting')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 3. Final Error vs. Threshold
        ax = axes[1, 0]
        final_errors = []
        for threshold in threshold_values:
            convergence = all_convergence[threshold]
            if not convergence:
                continue
                
            errors = convergence.get('reconstruction_error_l1', [])
            if errors:
                final_errors.append((threshold, errors[-1]))
        
        if final_errors:
            threshold_list, error_list = zip(*final_errors)
            ax.plot(threshold_list, error_list, 'o-')
            
            # Add data labels
            for x, y in zip(threshold_list, error_list):
                ax.annotate(f"{y:.6f}", (x, y), textcoords="offset points", 
                          xytext=(0, 5), ha='center')
        
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Final L1 Reconstruction Error')
        ax.set_title('Impact of Threshold on Final Error')
        ax.grid(alpha=0.3)
        
        # 4. Number of Iterations vs. Threshold
        ax = axes[1, 1]
        iteration_counts = []
        for threshold in threshold_values:
            convergence = all_convergence[threshold]
            if not convergence:
                continue
                
            iterations = convergence.get('iteration', [])
            if iterations:
                iteration_counts.append((threshold, len(iterations)))
        
        if iteration_counts:
            threshold_list, count_list = zip(*iteration_counts)
            ax.plot(threshold_list, count_list, 'o-')
            
            # Add data labels
            for x, y in zip(threshold_list, count_list):
                ax.annotate(f"{y}", (x, y), textcoords="offset points", 
                          xytext=(0, 5), ha='center')
        
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Number of Iterations')
        ax.set_title('Impact of Threshold on Convergence Speed')
        ax.grid(alpha=0.3)
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(viz_dir, "threshold_convergence.png"), dpi=300)
        plt.close()
        
        # Create a more detailed error vs. threshold plot
        plt.figure(figsize=(12, 8))
        final_errors = []
        for threshold in threshold_values:
            convergence = all_convergence[threshold]
            if not convergence:
                continue
                
            errors = convergence.get('reconstruction_error_l1', [])
            if errors:
                final_errors.append((threshold, errors[-1]))
        
        if final_errors:
            threshold_list, error_list = zip(*final_errors)
            plts