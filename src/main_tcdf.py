from src.tcdf.TCDF import *
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys
import glob


def runTCDF_multi(csv_files, **params):
    """Runs TCDF on multiple CSV files"""
    
    # Get all unique columns across all CSV files
    all_columns = set()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_columns.update(df.columns)
    
    first_df = pd.read_csv(csv_files[0])
    columns = [col for col in first_df.columns if col in all_columns]

    print(f'Features list: {len(columns)}')
    for idx, c in enumerate(columns):
        print(f'{idx} \t= {c} ')
    
    allcauses = dict()
    alldelays = dict()
    allreallosses = dict()
    allscores = dict()
    
    for c in columns:
        idx = columns.index(c)
        
        causes, causeswithdelay, realloss, scores = findcauses_multi(
            c, 
            cuda=params.get('cuda', False), 
            epochs=params.get('epochs', 2500), 
            kernel_size=params.get('kernel_size', 20), 
            layers=params.get('levels', 1), 
            log_interval=params.get('log_interval', 500), 
            lr=params.get('learning_rate', 0.001), 
            optimizername=params.get('optimizer', 'RMSprop'),
            seed=params.get('seed', 1111), 
            dilation_c=params.get('dilation_c', 20), 
            significance=params.get('significance', 0.8), 
            csv_files=csv_files
        )
        
        allscores[idx] = scores
        allcauses[idx] = causes
        alldelays.update(causeswithdelay)
        allreallosses[idx] = realloss
    
    return allcauses, alldelays, allreallosses, allscores, columns


def main(csv_files, params, save_path='log/'):
    """Main function that runs TCDF analysis on CSV files"""
    
    print(f"\nAnalyzing {len(csv_files)} CSV files...")
    
    allcauses, alldelays, allreallosses, allscores, columns = runTCDF_multi(
        csv_files, **params
    )
    
    print("==================================================================================")
    
    # Convert results to tigramite format
    tcdf_results = []
    for (source_idx, target_idx), lag in alldelays.items():
        source_var = columns[source_idx]
        target_var = columns[target_idx]
        
        # Determine link type
        link_type = "-->" if lag > 0 else "o-o"
        
        tcdf_results.append([source_var, target_var, lag, link_type, 1.0])

    # Create DataFrame
    df_tcdf = pd.DataFrame(tcdf_results, columns=["Variable i", "Variable j", "Time lag of i", "Link type i --- j", "Link value"])
    
    # Save CSV
    os.makedirs(str(save_path), exist_ok=True)
    df_tcdf.to_csv(os.path.join(save_path, 'tcdf_results.csv'), index=False)
    
    print("TCDF Results converted:")
    print(df_tcdf)
    
    return allcauses, alldelays, allreallosses, allscores, columns


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TCDF analysis')
    parser.add_argument('--csv_pattern', type=str, default='data/multi/*.csv', 
                       help='Pattern for CSV files')
    parser.add_argument('--kernel_size', type=int, default=20)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    parser.add_argument('--significance', type=float, default=0.8)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=True)
    
    args = parser.parse_args()
    
    csv_files = glob.glob(args.csv_pattern)
    
    params = {
        'kernel_size': args.kernel_size,
        'dilation_c': args.kernel_size,
        'levels': args.levels,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'significance': args.significance,
        'log_interval': args.log_interval,
        'seed': args.seed,
        'cuda': args.cuda
    }
    
    main(csv_files, params, args.plot)