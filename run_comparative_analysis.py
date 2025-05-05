#!/usr/bin/env python3
"""
Script to run a comparative analysis of graph summarization algorithms
on Amazon-0302, DBLP, and Email-Enron datasets.
"""

import os
import argparse
import sys
import networkx as nx
from comparative_analysis import ComparativeAnalysis

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run comparative analysis of graph summarization algorithms')
    
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--datasets', nargs='+', 
                        default=['Amazon-0302', 'DBLP', 'Email-Enron'],
                        help='Names of datasets to analyze')
    parser.add_argument('--k-ratios', nargs='+', type=float, default=[0.1, 0.2, 0.3],
                        help='Target k/n ratios to test')
    parser.add_argument('--max-runtime', type=int, default=600,
                        help='Maximum runtime in seconds for each algorithm run')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip creating visualizations')
    parser.add_argument('--skip-report', action='store_true',
                        help='Skip generating the comprehensive report')
    
    return parser.parse_args()

def main():
    """Main function to run the comparative analysis."""
    args = parse_arguments()
    
    # Initialize the analysis
    analysis = ComparativeAnalysis(output_dir=args.output_dir)
    
    # Load datasets
    datasets = analysis.load_datasets(dataset_names=args.datasets)
    
    if not datasets:
        print("Error: No datasets available for analysis.")
        sys.exit(1)
    
    # Compute baseline statistics
    analysis.compute_baseline_statistics(datasets)
    
    # Run comparative analysis
    analysis.run_comparative_analysis(
        datasets=datasets,
        k_ratios=args.k_ratios,
        max_runtime=args.max_runtime
    )
    
    # Create visualizations
    if not args.skip_visualizations:
        analysis.create_visualizations()
    
    # Identify implementation challenges
    analysis.identify_implementation_challenges()
    
    # Generate comprehensive report
    if not args.skip_report:
        report_path = analysis.generate_report()
        print(f"\nComparative analysis complete! Full report available at: {report_path}")
    else:
        print("\nComparative analysis complete!")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()