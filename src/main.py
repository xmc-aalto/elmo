import random
import numpy as np
import os
import torch
import hydra
from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.core.config_store import ConfigStore

from config import ( 
        PathWiki31K, PathEurlex4K, PathAmazonCat13K, PathAmazon670K, PathAmazonTitles670K, PathWiki500K, PathAmazon3M, 
        PathAmazonTitles3M,PathLFAmazonTitles131K, PathLFAmazon131K, PathLFWikiSeeAlso320K, PathLFWikiSeeAlsoTitles320K,
        PathLFAmazonTitles1P3M, PathMMAmazonTitles300K,SimpleConfig, validate_config,
        PathLFTitle2keywords, PathLFPaper2keywords
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
DATASET_NAME_MAP = {'eurlex4k': 'Eurlex-4K', 'amazoncat13k': 'AmazonCat-13K','wiki31k': 'Wiki10-31K', 'amazon670k': 'Amazon-670K',
                    'amazontitles670k':'AmazonTitles-670K','wiki500k': 'Wiki-500K','amazon3m':'Amazon-3M',
                    'amazontitles3m':'AmazonTitles-3M','lfamazon131k':'LF-Amazon-131K', 'lfamazontitles131k':'LF-AmazonTitles-131K',
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k', 'lfwikiseealsotitles320k':'LF-WikiSeeAlsoTitles-320k',
              'lfamazontitles1.3m':'LF-AmazonTitles-1.3M', 
              'lfpaper2keywords':'lfpaper2keywords', 'lftitle2keywords':'lftitle2keywords'}

# Path configuration based on the environment
ENVIRONMENT_TO_PATH = {'nasib-triton':'/scratch/work/nasibun1/projects/Datasets',
            'nasib-puhti':'/scratch/project_2001083/nasib/Datasets',
            'nasib-mahti':'/scratch/project_2001083/nasib/Datasets',
            'aalto-lab':'/l/WorkSpace/Datasets/XMC'}

# Dataset path mapping
DATASET_TO_PATH_OBJECT = {'eurlex4k': PathEurlex4K, 'amazoncat13k': PathAmazonCat13K, 'wiki31k': PathWiki31K, 'amazon670k': PathAmazon670K,
                          'amazontitles670k':PathAmazonTitles670K,'wiki500k' : PathWiki500K, 'amazon3m': PathAmazon3M,
                          'amazontitles3m':PathAmazonTitles3M,'lfamazon131k':PathLFAmazon131K,'lfamazontitles131k':PathLFAmazonTitles131K,
            'lfwikiseealso320k':PathLFWikiSeeAlso320K, 'lfwikiseealsotitles320k':PathLFWikiSeeAlsoTitles320K,
              'lfamazontitles1.3m':PathLFAmazonTitles1P3M,'mmamazontitles300k':PathMMAmazonTitles300K,
              'lfpaper2keywords': PathLFPaper2keywords, 'lftitle2keywords': PathLFTitle2keywords}

@hydra.main(version_base="1.2", config_path="../config/",config_name="config") 
def main(cfg: SimpleConfig ) -> None:
    '''
    Main function to initialize training process based on configuration.
    '''
    
    #print(cfg)

    # Set random seeds for reproducibility
    seed = cfg.dataset.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True

    
    
     # Determine dataset path
    dataset_path = cfg.dataset_path if os.path.exists(cfg.dataset_path) else ENVIRONMENT_TO_PATH[cfg.environment.running_env]
    path = DATASET_TO_PATH_OBJECT[cfg.dataset.data.dataset](dataset_path)

    
    #modify configuration
    if cfg.use_wandb:
        cfg.dataset.training.verbose.use_wandb = True
    if cfg.wandb_runname != "":
        cfg.dataset.training.verbose.wandb_runname = cfg.wandb_runname
    if cfg.log_fname != "":
        cfg.dataset.training.verbose.log_fname = cfg.log_fname
        
    if cfg.job_num is not None:

        cfg.dataset.training.verbose.wandb_runname += '_'+ str(cfg.job_num)

    #torch.cuda.memory._record_memory_history(enabled='all') # for memory snapshot
    
    #modify the config file
    # Modify the config file
    cfg2 = cfg.dataset
    with open_dict(cfg2):
        cfg2['environment'] = cfg['environment']
        #if cfg.job_num is not None:
        cfg2['jobnum'] = cfg.job_num
    cfg = cfg2
    
    #print(cfg)
    
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

    #print(f"hydra job num {cfg.job_num}")
    print(f"wandb run name: {cfg.training.verbose.wandb_runname }")
    
    runner.run_train(train_loader,test_loader,train_loader_eval)
    
    #runner.prepare_artifact_for_stage_two(test_loader,str(cfg.training.verbose.wandb_runname)+".pt")
    

if __name__ == '__main__':
    main()
