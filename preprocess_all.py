import os
import argparse
import time
from graph_preprocessor import GraphPreprocessor
from graph_format_converter import GraphFormatConverter

def main():
    parser = argparse.ArgumentParser(description='Preprocess graph datasets for SSumM algorithm and baselines')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing raw datasets')
    parser.add_argument('--processed_dir', type=str, default='./processed_data', help='Directory for processed data')
    parser.add_argument('--algorithm_dir', type=str, default='./algorithm_data', help='Directory for algorithm-specific data')
    parser.add_argument('--datasets', nargs='+', default=['DBLP', 'Amazon-0302', 'Email-Enron'], help='Datasets to process')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip_conversion', action='store_true', help='Skip algorithm format conversion step')
    parser.add_argument('--largest_cc_only', action='store_true', help='Extract only largest connected component')
    parser.add_argument('--skip_subsamples', action='store_true', help='Skip creation of subsamples')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Step 1: Preprocess graphs
    if not args.skip_preprocessing:
        print("\n=== Step 1: Preprocessing Graphs ===\n")
        processor = GraphPreprocessor(data_dir=args.data_dir, processed_dir=args.processed_dir)
        
        # Load datasets
        for dataset_name in args.datasets:
            try:
                processor.load_dataset(dataset_name, file_format="edge_list", delimiter=None, comment="#")
            except FileNotFoundError:
                print(f"WARNING: Dataset file not found for {dataset_name}. Skipping.")
        
        # Preprocess loaded datasets
        for dataset_name in list(processor.graphs.keys()):
            # Standard preprocessing
            processor.preprocess_graph(
                dataset_name,
                make_undirected=True,
                remove_self_loops=True,
                largest_cc_only=args.largest_cc_only,
                relabel_nodes=True
            )
            
            # Create subsamples
            if not args.skip_subsamples:
                processor.create_subsamples(
                    dataset_name,
                    sizes=[0.01, 0.05, 0.1, 0.25],
                    methods=["random_nodes", "snowball"]
                )
            
            # Analyze and save
            processor.analyze_graph(dataset_name)
            processor.save_processed_graph(dataset_name)
    
    # Step 2: Convert to algorithm-specific formats
    if not args.skip_conversion:
        print("\n=== Step 2: Converting to Algorithm-Specific Formats ===\n")
        converter = GraphFormatConverter(
            processed_dir=args.processed_dir, 
            output_dir=args.algorithm_dir
        )
        
        # Get all processed datasets (including subsamples)
        all_datasets = []
        for dataset_name in args.datasets:
            dataset_dir = os.path.join(args.processed_dir, dataset_name)
            if os.path.isdir(dataset_dir):
                all_datasets.append(dataset_name)
                
                # Add subsamples if they exist
                if not args.skip_subsamples:
                    subsample_pattern = f"{dataset_name}_"
                    subsamples = [d for d in os.listdir(args.processed_dir) 
                                  if os.path.isdir(os.path.join(args.processed_dir, d)) 
                                  and d.startswith(subsample_pattern)]
                    all_datasets.extend(subsamples)
        
        # Convert all datasets
        converter.convert_all_datasets(all_datasets)
        
        # Generate experiment configuration
        converter.generate_experiment_config(all_datasets)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n=== Preprocessing Complete ===")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Processed datasets: {', '.join(args.datasets)}")
    print(f"Processed data directory: {args.processed_dir}")
    print(f"Algorithm data directory: {args.algorithm_dir}")

if __name__ == "__main__":
    main()