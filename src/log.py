import datetime
import os
import torch
import wandb
from omegaconf import OmegaConf
import json

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
        logfile_path = f'./log' 
        if not os.path.exists(logfile_path ):
            os.mkdir(self.logfile_path)
        self.logfile_name = os.path.join(logfile_path,cfg.training.verbose.log_fname)  #.txtprint(self.logfile_name)
        self.json_path = os.path.join('Results',cfg.data.dataset,cfg.training.verbose.log_fname)
        os.makedirs(self.json_path, exist_ok=True)
        self.json_name = str(cfg.training.seed) + '_'+ str(cfg['jobnum'])+'.json' if cfg['jobnum'] is not None else str(cfg.training.seed) + '.json'
        self.json_name = os.path.join(self.json_path,self.json_name)
        #increment scale
        self.total_epoch = 0
        
        #loss logging
        self.iter_loss, self.epoch_loss = [], []
        
        #lr logger
        self.encoder_lr, self.meta_lr, self.xmc_lr = [], [], []
        
        #for performance logging
        self.trnp1, self.trnp3, self.trnp5 = [], [], []
        self.p1, self.p3, self.p5 = [], [], []
        self.psp1, self.psp3, self.psp5 = [], [], []
        self.psr1, self.psr3, self.psr5 = [], [], []
        self.ndcg1, self.ndcg3, self.ndcg5 = [], [], []
        self.r1, self.r3, self.r5 = [], [], []

        
        self.desc = []  
         
        #for gradient Logging
        self.grad_norm_xmc = []
        self.grad_norm_encoder = []
        self.grad_norm_combined = []
        self.grad_iter = [] # for wandb workaround
        self.trust_ratio_sparse = [] #norm/grad for layer (could be useful for large batch stability analysis)
        
        #memory logging
        self.mem, self.max_mem, self.model_memory = [], [], 0
        self.data = {"Config":{"data":OmegaConf.to_container(cfg.data),"model":OmegaConf.to_container(cfg.model),
                               "training":OmegaConf.to_container(cfg.training)},"Epoch Log":{}}
        self.logjson()
        
    def initialize_train(self,param_count):
        if self.cfg.training.verbose.logging:
            log_str = f"  Training model for Configuration \n -----Data Config------ \n {OmegaConf.to_yaml(self.cfg['data'])} \n  -----Model Config----- \
                \n {OmegaConf.to_yaml(self.cfg['model'])} \n ------Training Config------ \n {OmegaConf.to_yaml(self.cfg['training'])}"
            self.logfile(log_str)
            self.logfile(param_count)
        if self.cfg.training.verbose.use_wandb:
            #wandb.login('allow','')
            project_name = "LPXMC"
            self.run  = wandb.init(project=project_name,config=to_wandb_dict(self.cfg),name=self.cfg.training.verbose.wandb_runname)
                
        
    def logjson(self):
        with open(self.json_name, 'w') as file:
            # Write the updated data structure to the file
            json.dump(self.data, file, indent=4)

    def logfile(self, text):
        with open(self.logfile_name, 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')
         
            
    def naive_memory_logging(self,epoch):
        mem = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        max_mem = torch.cuda.max_memory_allocated(device=self.device)
        max_mem = round(max_mem / (1024 ** 3), 2)
        self.mem.append(mem)
        self.max_mem.append(max_mem)
        log_str = f' Memory allocted after trining epoch={epoch} is : {mem} GB \n'
        log_str += f' Peak Memory allocted after trining epoch={epoch} is : {max_mem} GB'
        self.logfile(log_str)
        if epoch==10 or epoch==20:
            self.logfile(torch.cuda.memory_summary())
            
    def model_memory_logging(self):
        max_mem = torch.cuda.max_memory_allocated(device=self.device)
        self.model_memory  = round(max_mem / (1024 ** 3), 2)
        self.data["model_memory"] = self.model_memory 
        self.logjson()
        
    def loss_logging(self,epoch,epoch_loss):
        self.epoch_loss.append(epoch_loss)
        log_str = f'   Epoch: {epoch+1:>2}   train_loss:{epoch_loss}'
        self.logfile(log_str)
        
    def test_perf_logging(self,epoch,metrics):
        
        tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
        self.p1.append(tp1)
        self.p3.append(tp3)
        self.p5.append(tp5)
        log_str = f'  Test set Performance : \n Epoch:{epoch+1:>2}, P@1:{tp1:.5f}   P@3:{tp3:.5f}  P@5:{tp5:.5f}'
        wandb_log = {'Test/P@1':tp1,'Test/P@3':tp3,'Test/P@5':tp5,'Train/Loss':self.epoch_loss[-1]}
        if self.cfg.training.evaluation.eval_psp:
            tpsp1, tpsp3, tpsp5 = metrics['PSP@K'][0], metrics['PSP@K'][2], metrics['PSP@K'][4]
            self.psp1.append(tpsp1)
            self.psp3.append(tpsp3)
            self.psp5.append(tpsp5)
            log_str += f' \n Epoch:{epoch+1:>2}, PSP@1:{tpsp1:.5f}   PSP@3:{tpsp3:.5f}  PSP@5:{tpsp5:.5f}'
            wandb_log.update({'Test/PSP@1':tpsp1,'Test/PSP@3':tpsp3,'Test/PSP@5':tpsp5})
        if self.cfg.training.evaluation.eval_recall:
            tr1, tr3, tr5 = metrics['R@K'][0], metrics['R@K'][2], metrics['R@K'][4]
            self.r1.append(tr1)
            self.r3.append(tr3)
            self.r5.append(tr5)
            log_str += f' \n Epoch:{epoch+1:>2}, R@1:{tr1:.5f}   R@3:{tr3:.5f}  R@5:{tr5:.5f}'
            wandb_log.update({'Test/R@1':tr1,'Test/R@3':tr3,'Test/R@5':tr5})
        if self.cfg.training.evaluation.eval_psr:
            tpsr1, tpsr3, tpsr5 = metrics['PSR@K'][0], metrics['PSR@K'][2], metrics['PSR@K'][4]
            log_str += f' \n Epoch:{epoch+1:>2}, PSR@1:{tpsr1:.5f}   PSR@3:{tpsr3:.5f}  PSR@5:{tpsr5:.5f}'
            wandb_log.update({'Test/PSR@1':tpsr1,'Test/PSR@3':tpsr3,'Test/PSR@5':tpsr3})
        if self.cfg.training.evaluation.eval_ndcg:
            tndcg1, tndcg3, tndcg5 = metrics['NDCG@K'][0], metrics['NDCG@K'][2], metrics['NDCG@K'][4]
            self.ndcg1.append(tndcg1)
            self.ndcg3.append(tndcg3)
            self.ndcg5.append(tndcg5)
            log_str += f' \n Epoch:{epoch+1:>2}, NDCG@1:{tndcg1:.5f}   NDCG@3:{tndcg3:.5f}  NDCG@5:{tndcg5:.5f}'
            wandb_log.update({'Test/NDCG@1':tndcg1,'Test/NDCG@3':tndcg3,'Test/NDCG@5':tndcg5})
        
            
        self.logfile(log_str)
            
        if self.cfg.training.verbose.use_wandb:
            wandb_log.update({'epoch':self.total_epoch})
            self.run.log(wandb_log)
            
        print(log_str)
    
    def train_perf_logging(self,epoch,metrics):
        trnp1, trnp3, trnp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
        self.trnp1.append(trnp1)
        self.trnp3.append(trnp3)
        self.trnp5.append(trnp5)
        print(f"Train set Performance: P@1:{trnp1:.5f}   P@3:{trnp3:.5f}   P@5:{trnp5:.5f}")
        log_str = f" Train set Performance :  Epoch:{epoch+1:>2}   P@1:{trnp1:.5f}   P@3:{trnp3:.5f}  P@5:{trnp5:.5f} "
        wandb_log = {'Train/P@1':trnp1,'Train/P@3':trnp3,'Train/P@5':trnp5}
        self.logfile(log_str)
        if self.cfg.training.verbose.use_wandb:
            wandb_log.update({'epoch':self.total_epoch})
            self.run.log(wandb_log)
            
    def lr_logger(self,lr_val,lr_val_xmc):
        wandb_log = {}

        wandb_log['LR_Encoder'] = lr_val[0]
        self.encoder_lr.append(lr_val[0])
        self.meta_lr.append(0)
        #if self.cfg.model.bottleneck.use_bottleneck_layer:
        wandb_log['LR_XMC']=lr_val_xmc[0]
        self.xmc_lr.append(lr_val_xmc[0])
        wandb_log.update({'epoch':self.total_epoch})
        if self.cfg.training.verbose.use_wandb:
            self.run.log(wandb_log)
            
    
    
    def step(self,epoch):
        
        r1 = self.r1[-1] if self.cfg.training.evaluation.eval_recall else 0
        r3 = self.r3[-1] if self.cfg.training.evaluation.eval_recall else 0
        r5 = self.r5[-1] if self.cfg.training.evaluation.eval_recall else 0
        
        ndcg1 = self.ndcg1[-1] if self.cfg.training.evaluation.eval_ndcg else 0
        ndcg3 = self.ndcg3[-1] if self.cfg.training.evaluation.eval_ndcg else 0
        ndcg5 = self.ndcg5[-1] if self.cfg.training.evaluation.eval_ndcg else 0
        
        trnp1 = self.trnp1[-1] if self.cfg.training.evaluation.train_evaluate else 0
        trnp3 = self.trnp3[-1] if self.cfg.training.evaluation.train_evaluate else 0
        trnp5 = self.trnp5[-1] if self.cfg.training.evaluation.train_evaluate else 0
        
        self.data["Epoch Log"].update({str(epoch):{"trn_loss":self.epoch_loss[-1],"test_P@k":[self.p1[-1],self.p3[-1],self.p5[-1]],
                        "test_PSP@K":[self.psp1[-1],self.psp3[-1],self.psp5[-1]],"test_NDCG@k":[ndcg1,ndcg3,ndcg5],
                        "test_R@k":[r1,r3,r5],
                        "train_P@k":[trnp1,trnp3,trnp5],
                        "LR_XMC":self.xmc_lr[-1],"LR_Encoder":self.encoder_lr[-1],
                        "memory":self.mem[-1],"peak_memory":self.max_mem[-1]}})
            
        
        self.logjson()
        self._reset_iter_states()
        
        
    def _reset_iter_states(self):
        self.iter_loss = []
        self.grad_norm_xmc = []
        self.grad_norm_encoder = []
        self.grad_norm_combined = []
        self.grad_iter = [] # for wandb workaround
        self.trust_ratio_sparse = [] #norm/grad for layer (could be useful for large batch stability analysis)
        
        
    def finalize(self):
        if self.cfg.training.verbose.use_wandb:
            wandb.finish()   
        

def to_wandb_dict(cfg):
    '''
    simple fix to create wandb dict
    
    '''
    wandb_cfg = {}
    #Data related config
    wandb_cfg['dataset'] = cfg.data.dataset
    wandb_cfg['augment_label_data'] = cfg.data.augment_label_data
    wandb_cfg['num_labels'] = cfg.data.num_labels
    wandb_cfg['max_len'] = cfg.data.max_len
    wandb_cfg['batch_size'] = cfg.data.batch_size
    
    #model related config
    wandb_cfg['encoder_model'] = cfg.model.encoder.encoder_model
    wandb_cfg['pool_mode'] = cfg.model.encoder.pool_mode
    wandb_cfg['feature_layers'] = cfg.model.encoder.feature_layers
    wandb_cfg['embed_dropout'] = cfg.model.encoder.embed_dropout
    wandb_cfg['use_ngame_encoder_weights'] = cfg.model.encoder.use_ngame_encoder_weights
    wandb_cfg['use_bottleneck_layer'] = cfg.model.bottleneck.use_bottleneck_layer
    wandb_cfg['bottleneck_size'] = cfg.model.bottleneck.bottleneck_size

    
    #Training related config
    wandb_cfg['loss_fn'] = cfg.training.loss_fn
    wandb_cfg['encoder_optimizer'] = cfg.training.encoder.optimizer
    wandb_cfg['xmc_optimizer'] = cfg.training.xmc.optimizer
    wandb_cfg['epochs'] = cfg.training.epochs
    wandb_cfg['grad_accum_step'] = cfg.training.encoder.grad_accum_step
    wandb_cfg['encoder_lr'] = cfg.training.encoder.lr
    wandb_cfg['lr'] = cfg.training.xmc.lr
    wandb_cfg['wd_encoder'] = cfg.training.encoder.wd
    wandb_cfg['warmup_steps'] = cfg.training.encoder.warmup_steps
    
    
    #Precision related Config #TODO 

    
    return wandb_cfg