import os
from dataclasses import dataclass, field
from omegaconf import DictConfig


@dataclass
class EnvironmentConfig:
    '''
    Configuration related to the running environment of the system.
    '''
    running_env: str = "aalto-lab"
    cuda_device_id: int = 0
    device: str = "cuda"

@dataclass
class DataConfig:
    '''
    Configuration for dataset specifics and data handling procedures.
    '''
    dataset: str = "lfamazon131k"
    is_lf_data: bool =  True
    augment_label_data: bool =  True
    use_filter_eval: bool = False
    num_labels: int = 131073
    max_len: int = 128
    num_workers: int = 8
    batch_size: int = 512
    test_batch_size: int = 512
    dtype: str = 'float32'
    label_sort: bool = False
    tokenized_loading: bool = False
    
@dataclass
class EncoderConfig:
    '''
    Configuration for the encoder model specifics.
    '''
    encoder_model: str =  "sentence-transformers/msmarco-distilbert-base-v4" #['sentence-transformers/all-roberta-large-v1','bert-base-uncased']
    encoder_tokenizer: str =  "sentence-transformers/msmarco-distilbert-base-v4"
    encoder_ftr_dim: int =  768
    pool_mode: str =  "last_hidden_avg" #[last_nhidden_conlast,last_hidden_avg]
    feature_layers: int =  1
    embed_dropout: float = 0.7
    use_torch_compile: bool = False
    use_ngame_encoder_weights: bool = False
    ngame_checkpoint: str = "./NGAME_ENCODERS/lfamazon131k/state_dict.pt"
    dtype: str = "float32"
   
@dataclass 
class bottleneckConfig:
    '''
    Configuration settings for the bottleneck layer, specifying
    whether to use the bottleneck layer, its size, and activation function.
    '''
    use_bottleneck_layer: bool =  True
    bottleneck_size: int = 4096
    bottleneck_activation: str = "relu"
    dtype: str = "float32"
  
@dataclass  
class XMCConfig:
    '''
    Configuration for the extreme layer.
    '''
    layer: str = "dense"
    input_features: int = 768
    output_features: int = 131073 #depends on num_labels in data
    use_torch_compile: bool = False
    dtype: str = "float32"
    implementation: str = "chunked"
    num_chunks: int = 2


@dataclass
class ModelConfig:
    '''
    Configuration grouping various model component settings.
    '''
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: bottleneckConfig = field(default_factory=bottleneckConfig)
    xmc: XMCConfig = field(default_factory=XMCConfig)
    
@dataclass
class PrecisionConfig:
    '''
    Configuration for Automatic Mixed Precision and Pure Low Precision Training.
    '''

    mode: str = "amp"  #[amp-automatic mixed precision,purelp-pure low precision (bf16 and lower),full(full precision, no amp or low precision)]
    purelp_dtype: str = "bfloat16" #[uniform low precision dtype]


@dataclass
class TEncoderConfig:
    lr: float = 1.0e-5  # learning rate
    wd: float = 0.001   # weight decay
    optimizer: str = 'adam'  # [adam,adamw,sgd]
    momentum: float = 0.9 # used with sgd 
    lr_scheduler: str = "CosineScheduleWithWarmup"   #[MultiStepLR,CosineScheduleWithWarmup,ReduceLROnPlateau]
    warmup_steps: int = 500
    grad_accum_step: int = 1
    implementation: str = "pytorch"  #[optimi,pytorch,custom]

@dataclass
class TXMCConfig:
    lr: float = 1.0e-4  # learning rate for head
    wd: float = 0.0001   # weight decay for head
    optimizer: str = "adam"  # [adam,adamw,sgd]
    momentum: float = 0 # used with sgd 
    lr_scheduler: str = "CosineScheduleWithWarmup"   #[MultiStepLR,CosineScheduleWithWarmup,ReduceLROnPlateau]
    warmup_steps: int = 500
    implementation: str = "pytorch"  #[optimi,pytorch,custom]

@dataclass
class TbottleneckConfig:
    lr: float = 1.0e-4
    wd: float = 0.001

@dataclass
class TFP8Config:
    fp8_embed: bool = True
    use_scaled_mm: bool = False
    use_fp8_encoder: bool = False
    fp8_encoder_delayed_scaling: bool = False
    head_ratio: float = 0.2
    using_head_kahan: bool = False
    debug_magnitude: bool = False
    debug_data_path: str = "./"
    num_split: int = 4
    

@dataclass
class EvaluationConfig:
    train_evaluate: bool = True
    train_evaluate_every: int = 10
    test_evaluate_every: int = 1
    A: float = 0.6  # for propensity calculation, epends on dataset
    B: float = 2.6  # for propensity calculation, epends on dataset
    eval_psp: bool = True
    eval_recall: bool = True
    eval_ndcg: bool = True
    eval_psr: bool = False

    
@dataclass
class VerboseConfig:
    show_iter: bool = False  # print loss during training
    print_iter: int = 2000  # how often (iteration) to print
    use_wandb: bool = False
    wandb_runname: str = "none"
    logging: bool = True
    log_fname: str = "log_amazontitles131k"
    use_checkpoint: bool = False  #whether to use automatic checkpoint
    best_p1: float = 0.462  # to store the model above this performance in case of automatic checkpoint
    
@dataclass
class exmyConfig:
    use_exmy: bool = False
    e: int = 8
    m: int = 7
    

@dataclass
class TrainingConfig:
    seed: int = 42
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    loss_fn: str = "positive_bce"
    epochs: int = 100
    training_steps: int = 1
    encoder: TEncoderConfig = field(default_factory=TEncoderConfig)
    xmc: TXMCConfig = field(default_factory=TXMCConfig)
    bottleneck: TbottleneckConfig = field(default_factory=TbottleneckConfig)
    FP8: TFP8Config = field(default_factory=TFP8Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    verbose: VerboseConfig = field(default_factory=VerboseConfig)
    use_checkpoint: bool = True  #whether to use automatic checkpoint
    checkpoint_file: str = "PBCE3"
    load_checkpoint_file: str = "None"
    best_p1: float = 0.50  # to store the model above this performance in case of automatic checkpoint
    grad_analysis: bool = False # whether to analyze gradient norms
    exmy: exmyConfig = field(default_factory=exmyConfig)

@dataclass
class DatasetConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    

@dataclass
class SimpleConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)



def validate_config(cfg: DictConfig):
    '''
    Validate the provided configuration to ensure it meets all requirements.
    '''
    # Example validation: check if a specific value meets a condition
    assert cfg.environment.device in ['cpu','cuda','cuda:0'], " Unknown device Selected"
    if 'lf' not in cfg.data.dataset and cfg.data.augment_label_data:
        raise ValueError("Can't Augment Label data for non label feature dataset. make augment_label_data=False or change the dataset")
    if 'lf' not in cfg.data.dataset and cfg.data.use_filter_eval:
        raise ValueError(" Can't use Filter evaluation for Non LF datasets. No reciprocal pairs.")
    if cfg.model.encoder.pool_mode=='last_hidden_avg' and not cfg.model.encoder.feature_layers==1:
            raise ValueError('The selected Pooling mode should have feature_layers=1')
    if cfg.training.encoder.grad_accum_step==0:
        print('grad acculation should not be zero. setting it to 1')
        cfg.training.encoder.grad_accum_step = 1
    print('All Validation passed..')
    
        
##-------------------------------------------------------------------------------------------------------------------##
##---------------------------------------Dataset Paths Configuration-------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------##
    
class PathWiki31K:
    '''
    Wiki10-31K Dataset.
    '''
    def __init__(self,root_path):
        
        self.root_folder = 'Wiki10-31K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_path = os.path.join(self.dataset_path,'BOW')
        self.bow_train_path = os.path.join(self.bow_path,'train.txt')
        self.bow_test_path = os.path.join(self.bow_path,'test.txt')
        self.bowXf_path = os.path.join(self.bow_path,'Xf.txt')
        self.bowY_path = os.path.join(self.bow_path,'Y.txt')


class PathEurlex4K:
    '''
    EurLex-4K Dataset.
    
    '''
    def __init__(self,root_path):

        self.root_folder = 'Eurlex-4K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')

class PathAmazonCat13K:
    '''
    AmazonCat-13K Dataset.
    
    '''
    def __init__(self,root_path):
        
        self.root_folder = 'AmazonCat-13K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        


class PathAmazon670K:
    '''
    Amazon-670K Dataset
    '''
    def __init__(self,root_path):

        self.root_folder = 'Amazon-670K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train_v1.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')


class PathAmazonTitles670K:
    '''
    AmazonTitles-670K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'AmazonTitles-670K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.train_json = os.path.join(self.dataset_path,'trn.json')
        self.test_json = os.path.join(self.dataset_path,'tst.json')
        #self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        #self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'trn_X_Xf.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'tst_X_Xf.txt')



class PathWiki500K:
    '''
    Wiki-500K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'Wiki-500K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')

class PathAmazon3M:
    '''
    Amazon-3M Dataset.
    
    '''
    def __init__(self,root_path):

        self.root_folder = 'Amazon-3M'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train_v1.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')
        
        
class PathAmazonTitles3M:
    '''
    AmazonTitles-670K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'AmazonTitles-3M'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.train_json = os.path.join(self.dataset_path,'trn.json')
        self.test_json = os.path.join(self.dataset_path,'tst.json')
        #self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        #self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'trn_X_Xf.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'tst_X_Xf.txt')
        
##----------------------------------------------- Label Features Datasets-----------------------###
##-------------------------------------------------------------------------------------------------##

class PathLFAmazonTitles131K:
    '''
    LF-AmazonTitles-131K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-AmazonTitles-131K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')
        
        
class PathMMAmazonTitles300K:
    '''
    MM-AmazonTitles-300K Dataset. (multimodal)
    
    '''
    def __init__(self,root_path):
        
        self.root_folder = 'MM-AmazonTitles-300K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_data')
        self.train_json = os.path.join(self.raw_text_path,'train.json')
        self.test_json = os.path.join(self.raw_text_path,'test.json')
        self.label_json = os.path.join(self.raw_text_path,'label.json')
        self.filter_labels_train = os.path.join(self.dataset_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.dataset_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,os.path.join('BOW','train.txt'))
        self.bow_test_path = os.path.join(self.dataset_path,os.path.join('BOW','test.txt'))
        
        #Caption files (from BLIP-2)
        self.train_caption_path = os.path.join(self.raw_text_path,'train_caption.json')
        self.test_caption_path = os.path.join(self.raw_text_path,'test_caption.json')
        self.label_caption_path = os.path.join(self.raw_text_path,'label_caption.json')
        
        
class PathLFAmazon131K:
    '''
    LF-Amazon-131K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-Amazon-131K'
        self.dataset_path = os.path.join(root_path,self.root_folder)
        #Raw text
        self.raw_text_path = self.dataset_path  # os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')
        

class PathLFWikiSeeAlso320K:
    '''
    LF-WikiSeeAlso-320K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-WikiSeeAlso-320K'
        self.dataset_path = os.path.join(root_path,self.root_folder)
        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')


class PathLFPaper2keywords:
    '''
    LF-Paper2keywords Dataset.
    '''
    def __init__(self, root_path):

        self.root_folder = 'lfpaper2keywords'
        self.dataset_path = os.path.join(root_path, self.root_folder)
        #Raw text
        self.train_json = os.path.join(self.dataset_path, 'trn.json')
        self.test_json = os.path.join(self.dataset_path, 'tst.json')
        self.label_json = os.path.join(self.dataset_path, 'lbl.json')


class PathLFTitle2keywords:
    '''
    LF-Title2keywords Dataset.
    '''
    def __init__(self, root_path):

        self.root_folder = 'lftitle2keywords'
        self.dataset_path = os.path.join(root_path, self.root_folder)
        #Raw text
        self.train_json = os.path.join(self.dataset_path, 'trn.json')
        self.test_json = os.path.join(self.dataset_path, 'tst.json')
        self.label_json = os.path.join(self.dataset_path, 'lbl.json')

class PathLFWikiSeeAlsoTitles320K:
    '''
    LF-WikiSeeAlsoTitles-320K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-WikiSeeAlsoTitles-320K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')


class PathLFAmazonTitles1P3M:
    def __init__(self,root_path):

        self.root_folder = 'LF-AmazonTitles-1.3M'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')