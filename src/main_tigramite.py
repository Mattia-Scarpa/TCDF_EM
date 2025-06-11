import os
import sys
import numpy as np
import pandas as pd
import glob
import pickle as pkl

# Add tigramite path if needed
# sys.path.append('/path/to/tigramite/')

from . import tigramite
from .tigramite import data_processing as pp
from .tigramite import plotting as tp
from .tigramite.pcmci import PCMCI
from .tigramite.lpcmci import LPCMCI

from .tigramite.independence_tests.parcorr import ParCorr
from .tigramite.independence_tests.robust_parcorr import RobustParCorr
from .tigramite.independence_tests.parcorr_wls import ParCorrWLS 
from .tigramite.independence_tests.gpdc import GPDC
from .tigramite.independence_tests.cmiknn import CMIknn
from .tigramite.independence_tests.cmisymb import CMIsymb
from .tigramite.independence_tests.gsquared import Gsquared
from .tigramite.independence_tests.regressionCI import RegressionCI


def load_multiple_csv_files(csv_files):
    """Load and combine multiple CSV files into a single dataset"""
    
    if len(csv_files) == 1:
        # Single file
        df = pd.read_csv(csv_files[0])
    else:
        # Multiple files - concatenate them
        dataframes = []
        for file in csv_files:
            df_temp = pd.read_csv(file)
            dataframes.append(df_temp)
        df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    return df


def setup_conditional_independence_test(test_name, **test_params):
    """Setup the conditional independence test based on parameters"""
    
    test_mapping = {
        'parcorr': ParCorr,
        'robust_parcorr': RobustParCorr,
        'parcorr_wls': ParCorrWLS,
        'gpdc': GPDC,
        'cmiknn': CMIknn,
        'cmisymb': CMIsymb,
        'gsquared': Gsquared,
        'regressionci': RegressionCI
    }
    
    if test_name.lower() not in test_mapping:
        raise ValueError(f"Test {test_name} not supported. Available tests: {list(test_mapping.keys())}")
    
    test_class = test_mapping[test_name.lower()]
    return test_class(**test_params)


def run_pcmci_analysis(dataframe, cond_ind_test, **pcmci_params):
    """Run PCMCI analysis with given parameters"""
    
    # Extract parameters
    verbosity = pcmci_params.get('verbosity', 1)
    tau_max = pcmci_params.get('tau_max', 8)
    tau_min = pcmci_params.get('tau_min', 0)
    pc_alpha = pcmci_params.get('pc_alpha', 0.05)
    alpha_level = pcmci_params.get('alpha_level', 0.01)
    fdr_method = pcmci_params.get('fdr_method', 'fdr_bh')
    max_conds_dim = pcmci_params.get('max_conds_dim', None)
    max_combinations = pcmci_params.get('max_combinations', 1)
    
    # Create PCMCI object
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=cond_ind_test,
        verbosity=verbosity
    )
    
    # Get lagged dependencies if requested
    if pcmci_params.get('get_lagged_dependencies', True):
        correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']
    else:
        correlations = None
    
    # Run PCMCI
    results = pcmci.run_pcmci(
        tau_max=tau_max, 
        pc_alpha=pc_alpha, 
        alpha_level=alpha_level,
        tau_min=tau_min,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations
    )
    
    # Get corrected p-values
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results['p_matrix'], 
        tau_max=tau_max, 
        fdr_method=fdr_method
    )
    
    # Print significant links
    if verbosity > 0:
        pcmci.print_significant_links(
            p_matrix=q_matrix,
            val_matrix=results['val_matrix'],
            alpha_level=alpha_level
        )
    
    # Get graph
    graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix, 
        alpha_level=alpha_level, 
        tau_min=tau_min, 
        tau_max=tau_max, 
        link_assumptions=None
    )
    
    # Add additional results
    results['graph'] = graph
    results['q_matrix'] = q_matrix
    results['correlations'] = correlations
    results['pcmci_object'] = pcmci
    
    return results


def save_results(results, var_names, save_path, filename_prefix='tigramite'):
    """Save PCMCI results to files"""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
    
    # Save main results to CSV
    csv_filename = os.path.join(save_path, f'results_{filename_prefix}.csv')
    tp.write_csv(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        save_name=csv_filename,
        digits=5,
    )
    
    # Save complete results to pickle
    pickle_filename = os.path.join(save_path, 'data', f'results_{filename_prefix}.pkl')
    # Remove pcmci_object before saving to avoid serialization issues
    results_to_save = {k: v for k, v in results.items() if k != 'pcmci_object'}
    with open(pickle_filename, 'wb') as f:
        pkl.dump(results_to_save, f)
    
    print(f"Results saved to: {csv_filename}")
    print(f"Complete results saved to: {pickle_filename}")
    
    return csv_filename, pickle_filename


def main(csv_files, tigramite_params, save_path='log/', plot_results=False):
    """Main function to run Tigramite analysis"""
    
    print(f"\nAnalyzing {len(csv_files)} CSV files with TIGRAMITE...")
    
    # Set random seed if provided
    seed = tigramite_params.get('seed', 42)
    np.random.seed(seed)
    
    # Load data
    df = load_multiple_csv_files(csv_files)
    print(f"Loaded data with shape: {df.shape}")
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    var_names = list(df.columns)
    
    # Create tigramite dataframe
    dataframe = pp.DataFrame(
        data, 
        datatime={0: np.arange(len(data))}, 
        var_names=var_names
    )
    
    print(f"Variables: {var_names}")
    
    # Setup conditional independence test
    test_name = tigramite_params.get('cond_ind_test', 'parcorr')
    test_params = tigramite_params.get('test_params', {'significance': 'analytic'})
    cond_ind_test = setup_conditional_independence_test(test_name, **test_params)
    
    # Extract PCMCI parameters
    pcmci_params = {k: v for k, v in tigramite_params.items() 
                   if k not in ['cond_ind_test', 'test_params', 'seed']}
    
    # Run PCMCI analysis
    print(f"\nRunning PCMCI with {test_name} test...")
    results = run_pcmci_analysis(dataframe, cond_ind_test, **pcmci_params)
    
    # Save results
    csv_file, pickle_file = save_results(results, var_names, save_path)
    
    print(f"\nAnalysis completed!")
    print(f"Found {np.sum(results['graph'] != '')} significant links")
    
    if plot_results and hasattr(tp, 'plot_graph'):
        try:
            tp.plot_graph(
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=var_names
            )
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    return results, var_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Tigramite PCMCI analysis')
    parser.add_argument('--csv_pattern', type=str, default='data/multi/*.csv', 
                       help='Pattern for CSV files')
    parser.add_argument('--cond_ind_test', type=str, default='parcorr',
                       help='Conditional independence test')
    parser.add_argument('--tau_max', type=int, default=8,
                       help='Maximum time lag')
    parser.add_argument('--tau_min', type=int, default=0,
                       help='Minimum time lag')
    parser.add_argument('--alpha_level', type=float, default=0.01,
                       help='Significance level')
    parser.add_argument('--verbosity', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_path', type=str, default='log/',
                       help='Path to save results')
    parser.add_argument('--plot', type=bool, default=False,
                       help='Whether to plot results')
    
    args = parser.parse_args()
    
    csv_files = glob.glob(args.csv_pattern)
    
    tigramite_params = {
        'cond_ind_test': args.cond_ind_test,
        'test_params': {'significance': 'analytic'},
        'tau_max': args.tau_max,
        'tau_min': args.tau_min,
        'alpha_level': args.alpha_level,
        'verbosity': args.verbosity,
        'seed': args.seed,
        'get_lagged_dependencies': True,
        'fdr_method': 'fdr_bh',
        'pc_alpha': None,
        'max_conds_dim': None,
        'max_combinations': 1
    }
    
    main(csv_files, tigramite_params, args.save_path, args.plot)