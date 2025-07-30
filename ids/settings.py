import argparse
    

def gather_settings():
    # Training settings
    parser = argparse.ArgumentParser(description='Train IDS and refine it with AG')
    
    # AG parameters
    parser.add_argument('--ag', type=str, required=False, default='alertNetAG',
                        help='attack graph file', choices=['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG'])
    parser.add_argument('--ag_path_prob', type=float, required=False, default=0.0,
                        help='probability for a random attack path to be activated in the AG')
    
    # Dataset parameters
    parser.add_argument('--train_percentage', type=float, required=False, default=0.75,
                        help='percentage of data of cicids to be used for training')
    parser.add_argument('--features_mode', type=str, required=False, default='all', #'top_10',
                        help='top_k features to be used to train the IDS model')

    # IDS parameters 
    parser.add_argument('--dt_depth', type=int, required=False, default=5,
                        help='maximum depth of the decision tree IDS')
    parser.add_argument('--min_samples_split', type=int, required=False, default=2,
                        help='minimum number of samples required to split an internal node of the decision tree IDS')
    parser.add_argument('--min_samples_leaf', type=int, required=False, default=1,
                        help='minimum number of samples required to be at a leaf node')
    
    # Folders parameters
    parser.add_argument('--log_folder', type=str, default='logs')
    parser.add_argument('--models_folder', type=str, default='ckpts')
    parser.add_argument('--metrics_folder', type=str, default='metrics')
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--ags_folder', type=str, default='data/ags')
    parser.add_argument('--csvs_folder', type=str, default='data/csvs')
    

    settings = parser.parse_args()
    return settings
