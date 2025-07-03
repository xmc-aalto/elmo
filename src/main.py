import random
import numpy as np
import os
import torch
import hydra
from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.core.config_store import ConfigStore

from config import ( 
        PathAmazon670K, PathAmazonTitles670K, PathWiki500K, PathAmazon3M, 
        PathLFAmazonTitles131K, PathLFWikiSeeAlso320K, PathLFAmazonTitles1P3M, 
        SimpleConfig, validate_config, PathLFPaper2keywords
)
from data import DataHandler
from runner import Runner

# Register resolvers for configuration variables based on conditional logic
OmegaConf.register_new_resolver(
    'encoder_feature_size',
    lambda enc_name: 1024 if 'large' in enc_name else 768
)
OmegaConf.register_new_resolver(
    'input_size_select',
    lambda use_bottleneck, bottleneck_size, feature_layers, feature_dim: (
        bottleneck_size if use_bottleneck else feature_layers * feature_dim
    )
)

# Initialize configuration store
cs = ConfigStore.instance()
cs.store(name="simple_config", node=SimpleConfig)

# Map dataset names to user-friendly names
DATASET_NAME_MAP = {'amazon670k': 'Amazon-670K',
                    'amazontitles670k':'AmazonTitles-670K','wiki500k': 'Wiki-500K','amazon3m':'Amazon-3M',
                    'lfamazontitles131k':'LF-AmazonTitles-131K',
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k', 
              'lfamazontitles1.3m':'LF-AmazonTitles-1.3M', 
              'lfpaper2keywords':'lfpaper2keywords'}

# Path configuration based on the environment
ENVIRONMENT_TO_PATH = {'guest':'Datasets'}

# Dataset path mapping
DATASET_TO_PATH_OBJECT = {'amazon670k': PathAmazon670K,
                          'amazontitles670k':PathAmazonTitles670K,'wiki500k' : PathWiki500K, 'amazon3m': PathAmazon3M,
                          'lfamazontitles131k':PathLFAmazonTitles131K, 'lfwikiseealso320k':PathLFWikiSeeAlso320K,
              'lfamazontitles1.3m':PathLFAmazonTitles1P3M,'lfpaper2keywords': PathLFPaper2keywords}

@hydra.main(version_base="1.2", config_path="../config/",config_name="config") 
def main(cfg: SimpleConfig ) -> None:
    '''
    Main function to initialize training process based on configuration.
    '''

    # Set random seeds for reproducibility
    seed = cfg.dataset.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    
    # Determine dataset path
    dataset_path = cfg.dataset_path if os.path.exists(cfg.dataset_path) else ENVIRONMENT_TO_PATH[cfg.environment.running_env]
    path = DATASET_TO_PATH_OBJECT[cfg.dataset.data.dataset](dataset_path)

    
    #override configuration based on config/config.yaml
    if cfg.use_wandb:
        cfg.dataset.training.verbose.use_wandb = True
    if cfg.wandb_runname != "":
        cfg.dataset.training.verbose.wandb_runname = cfg.wandb_runname
    if cfg.log_fname != "":
        cfg.dataset.training.verbose.log_fname = cfg.log_fname
        
    if cfg.job_num is not None:

        cfg.dataset.training.verbose.wandb_runname += '_'+ str(cfg.job_num)

    
    # Modify the config file
    cfg2 = cfg.dataset
    with open_dict(cfg2):
        cfg2['environment'] = cfg['environment']
        cfg2['jobnum'] = cfg.job_num
    cfg = cfg2
    
    
    print('Initializing data handler and data loading')
    
    data_handler = DataHandler(cfg,path)
    train_dset, test_dset, train_dset_eval = data_handler.getDatasets()
    train_loader_eval = data_handler.getDataLoader(train_dset_eval,mode='test')
    test_loader = data_handler.getDataLoader(test_dset,mode='test')
    train_loader = data_handler.getDataLoader(train_dset,mode='train')
    
    # Validate and adjust configurations dynamically
    cfg.training.training_steps = len(train_loader)
    validate_config(cfg)

    runner = Runner(cfg,path,data_handler)  
    
    #Training
    runner.run_train(train_loader,test_loader,train_loader_eval)
    

if __name__ == '__main__':
    main()
