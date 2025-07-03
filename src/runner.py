import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch.optim import Adam,AdamW,SGD
from optimi import AdamW as CAdamW
from optimi import Adam as CAdam
from optimi import SGD as CSGD
from optimi import Adam as CAdam
import transformers

from log import Logger
from model import SimpleTModel
from evaluate import Evaluator
from utils import print_trainable_parameters, EarlyStopping
from evaluate import sparsification



dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32}
optimizer_map_optimi = {'adam':CAdam,'adamw':CAdamW,'sgd':CSGD}
optimizer_map_torch = {'adam':Adam,'adamw':AdamW,'sgd':SGD}
    

class Runner:
    
    def __init__(self,cfg,path,data_handler):
        
        self.cfg = cfg
        self.path = path
        self.label_map = data_handler.label_map
        self.device = torch.device(cfg.environment.device)
        self._initialize_settings(cfg,data_handler) 
        self.running_evaluation = cfg.training.evaluation.running_evaluation

        #Logging Object
        if cfg.training.verbose.logging:
            self.LOG = Logger(cfg)
        
        #keep tracking of total iterations and epochs 
        self.total_iter = 1
        self.total_epoch = 1
        
        
    def _initialize_settings(self,cfg,data_handler):
        
        self.model = SimpleTModel(cfg,self.path)
        param_list_encoder = self.model.param_list()
        self.param_count = print_trainable_parameters(self.model)
        
        # Following Renee practice NGAME M1 encoder is used for short-text titles datasets 
        if cfg.model.encoder.use_ngame_encoder_weights:
            self._load_ngame_encoder()
        

        self.optimizer_encoder = self._get_optimizer(cfg,param_list_encoder,cfg.training.encoder.optimizer,
                                                     cfg.training.encoder.implementation,cfg.training.encoder.lr,cfg.training.encoder.momentum)
        
        # dummy single parameter optimizer to use existing LR scheduler
        self.optimizer_xmc = torch.optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=cfg.training.xmc.lr)

            
        self.lr_scheduler_encoder = self._get_lr_scheduler(self.optimizer_encoder,cfg.training.encoder.lr_scheduler,
                                                           cfg.training.encoder.warmup_steps,min_lr=2.0e-6)
        self.lr_scheduler_xmc = self._get_lr_scheduler(self.optimizer_xmc,cfg.training.xmc.lr_scheduler,
                                                       cfg.training.xmc.warmup_steps,min_lr=0.001)
        

        self.early_stopping = EarlyStopping(patience=25,delta=5e-5,mode='max')
        
    
        #Evaluators initialization
        self.test_evaluator_me = Evaluator(cfg,data_handler.label_map,data_handler.test_labels) #memory efficient version but supports only P@K
        self.test_evaluator_d = Evaluator(cfg,data_handler.label_map,data_handler.test_labels,train_labels=data_handler.train_labels,
                                          mode='debug',filter_path=self.path.filter_labels_test if cfg.data.use_filter_eval else None)
        
    
    def _get_optimizer(self,cfg,param_list,optimizer,implementation,lr,m):
        momentum_value = m
        kahan=True
        if optimizer not in ['adam','adamw']:
            kahan=False
            
            if isinstance(param_list,list):
                for param_group in param_list:
                    param_group['momentum'] = momentum_value
            else:
                param_list['momentum'] = momentum_value
        
        #check the implementation now
        if implementation in ['pytorch']:
            return optimizer_map_torch[optimizer](param_list)
        elif implementation in ['optimi']:
            return optimizer_map_optimi[optimizer](param_list,lr=lr,kahan_sum=kahan,weight_decay=self.cfg.training.encoder.wd) #,foreach=False
        else:
            raise ValueError('Only two optimizer implementation modes supported now')


    
    def _get_lr_scheduler(self, optimizer,scheduler_type,warmup_steps,min_lr=1.0e-5):
        
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                        factor=0.1, patience=2,eps=1e-4, min_lr=min_lr, verbose=True)
        elif scheduler_type == "MultiStepLR":
            #epochs = self.cfg.training.epochs
            total_steps = self.cfg.training.training_steps*self.cfg.training.epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,gamma=0.15,milestones=[int(total_steps/ 2), int(total_steps * 3 / 4)],last_epoch=-1)
        elif scheduler_type == "CosineScheduleWithWarmup":
            total_steps = self.cfg.training.training_steps*self.cfg.training.epochs
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
        elif scheduler_type == "ConstantLR":
            scheduler = transformers.get_constant_schedule(optimizer)
        else:
            raise ValueError(f"Unsupported scheduler type {scheduler_type}")
        return scheduler
        
    
    def run_one_epoch(self,epoch: int,train_loader) -> int:
        '''
        Train model for one epoch.
        Args:
            epoch: current epoch number
            train_loader: dataloader for the train set
        
        '''
        epoch_loss = 0
        loss_nprint = 1
        loss_print_every = 8000000 # if > num_iter, never calculate and print the loss (use skip loss)
        bar = tqdm(total=len(train_loader))
        bar.set_description(f'{epoch}')
        self.model.zero_grad()
        self.model.train()
        
        for i,data in enumerate(train_loader):

            tokens,mask,labels = data
            tokens, mask, labels = tokens.to(self.device),mask.to(self.device), labels.to(self.device)
            bsz = tokens.shape[0]

            out = self.model(tokens,mask)
            
            if (i+1)% loss_print_every==0:
                loss = self.model.xfc.xfc_forward_backward(out,labels,self.lr_scheduler_xmc.get_last_lr(),skip_loss=False)
                epoch_loss += loss
                loss_nprint += 1
            else:
                loss = self.model.xfc.xfc_forward_backward(out,labels,self.lr_scheduler_xmc.get_last_lr(),skip_loss=True)
                epoch_loss += loss
                loss_nprint += 1

            #grad accumulation on encoder
            self.model.xfc.grad_input[0:bsz, :].div_(self.cfg.training.encoder.grad_accum_step) 
            #autograd engine for encoder backward pass 
            out.backward(self.model.xfc.grad_input[0:bsz,:]) 
            
            
            if (i+1) % self.cfg.training.encoder.grad_accum_step==0:
                self.optimizer_encoder.step()
                self.optimizer_encoder.zero_grad(set_to_none=True)
                
            self.lr_scheduler_encoder.step()
            self.lr_scheduler_xmc.step() # XMC update is independent of grad_accum_step
            
            if self.cfg.training.verbose.logging:
                self.LOG.iter_loss.append(loss)

            self.total_iter +=1
            bar.update(1)
            bar.set_postfix(loss=epoch_loss)  
                

        return epoch_loss/ loss_nprint
        
        
    def run_train(self,train_loader,test_loader,train_loader_eval=None):

        self.best_p1 = self.cfg.training.best_p1

        print('Training Started for a Single Configuration...')
        print(f"Data Config: {self.cfg['data']} Model Config: {self.cfg['model']}  Training Config: {self.cfg['training']}")
        
        if self.cfg.training.verbose.logging:
            self.LOG.initialize_train(self.param_count)
        
        if self.cfg.training.verbose.logging:
            self.LOG.model_memory_logging()

        for epoch in range(self.cfg.training.epochs):
                
            epoch_loss = self.run_one_epoch(epoch,train_loader)
            print(f'Epoch:{epoch+1}   Epoch Loss: {epoch_loss:.7f}')
            
            if self.cfg.training.verbose.logging:
                self.LOG.loss_logging(epoch,epoch_loss)
                self.LOG.naive_memory_logging(epoch)

            #Test Evaluation
            if epoch % self.cfg.training.evaluation.test_evaluate_every==0 and self.running_evaluation:
                metrics  = self.test_evaluator_d.Calculate_Metrics(test_loader,self.model)
                if self.cfg.training.verbose.logging:
                    self.LOG.test_perf_logging(epoch,metrics)
                tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]

                if tp1>self.cfg.training.best_p1 and self.cfg.training.use_checkpoint:
                    self.best_p1 = tp1
                    temp = 'LP' 
                    temp = temp +'_' + self.cfg.training.checkpoint_file
                    if not os.path.exists(f'models/{self.cfg.data.dataset}/'):
                        os.makedirs(f'models/{self.cfg.data.dataset}/')
                    name = f'models/{self.cfg.data.dataset}/{self.cfg.data.dataset}_{temp}_best_test.pt'
                    self.save_checkpoint(epoch,name)
            
            #Training evaluation 
            if epoch % self.cfg.training.evaluation.train_evaluate_every==0 and self.cfg.training.evaluation.train_evaluate and self.running_evaluation:
                metrics  = self.test_evaluator_me.Calculate_Metrics(train_loader_eval,self.model)
                if self.cfg.training.verbose.logging:
                    self.LOG.train_perf_logging(epoch,metrics)
 

            if self.running_evaluation:
                self.early_stopping(tp3)
            
            if self.early_stopping.early_stop:
                print("Early stopping!")
                break
    
            self.total_epoch += 1
            
            #write the learning rates to wandb
            if self.cfg.training.verbose.logging and self.running_evaluation:
                self.LOG.total_epoch += 1
                self.LOG.lr_logger(self.lr_scheduler_encoder.get_last_lr(),self.lr_scheduler_xmc.get_last_lr())
                self.LOG.step(epoch)
                
        print('Training Finished.')
        if self.cfg.training.verbose.logging:
            self.LOG.finalize()

    def save_checkpoint(self,epoch,name):
        checkpoint = {
            'config':self.cfg,
            'state_dict_encoder': self.model.state_dict(),
            'xmc_weight': self.model.xfc.xfc_weight,
            'optimizer': self.optimizer_encoder.state_dict(),
            'epoch': epoch,
            'label_map':self.label_map
        }
        torch.save(checkpoint, name)

    def load_checkpoint(self,name,train_loader,test_loader,train_loader_eval):
        checkpoint = torch.load(name)
        try:
            self.model.load_state_dict(checkpoint['state_dict_encoder'], strict=False)
            self.model.xfc.xfc_weight = checkpoint['xmc_weight']
        except RuntimeError as E:
            print(E)

        self.optimizer_encoder.load_state_dict(checkpoint['optimizer'])
        self.cfg = checkpoint['config']
        self.label_map = checkpoint['label_map']
        train_loader.dataset.label_map = checkpoint['label_map']
        test_loader.dataset.label_map = checkpoint['label_map']
        train_loader_eval.dataset.label_map = checkpoint['label_map']

        return checkpoint['epoch']
    
    def prepare_artifact_for_stage_two(self,data_loader,name):
        features, labels = self.feature_extraction(data_loader)
        checkpoint = {
            'config':self.cfg,
            'xmc_weight': self.model.xfc.xfc_weight.to('cpu').detach().float(),
            'features': features,
            'labels': labels,
            'label_map':self.label_map
        }
        torch.save(checkpoint, name)

                    
    def feature_extraction(self, data_loader):
        """
        Feature extraction from encoder or bottleneck if activated.
        Make sure shuffle is off in the data_loader.
        
        Args:
        - model: Model for generating predictions.
        - data_loader: DataLoader providing batches of data (tokens, masks, labels).
        - cfg: Configuration object with attributes like num_labels and device.
        
        Returns:
        - A dense numpy array of predictions with shape (N, num_labels), where N is the dataset size.
        """
        self.model.eval()
        num_samples = len(data_loader.dataset)
        embedding_dim = self.cfg.model.encoder.encoder_ftr_dim  # Replace with actual dimension
        pred_matrix = np.zeros((num_samples, embedding_dim))
        current_index = 0
        print(f'Preparing features for the second stage training... for dataset:{data_loader.dataset}')
        with torch.no_grad():
            for data in tqdm(data_loader):
                tokens, mask, _ = data
                tokens, mask = tokens.to(self.device), mask.to(self.device)
                embed = self.model(tokens, mask)
                batch_size = tokens.size(0)
                pred_matrix[current_index:current_index + batch_size] = embed.cpu().float().numpy()
                current_index += batch_size
                
        gt_labels = sparsification(data_loader.dataset.labels,self.label_map,self.cfg.data.num_labels)

        return pred_matrix, gt_labels
    
    
    def _load_ngame_encoder(self):
        path_to_ngame_model = self.cfg.model.encoder.ngame_checkpoint
        print("Using NGAME pretrained encoder. Loading from {}".format(path_to_ngame_model))
        new_state_dict = OrderedDict()
        old_state_dict = torch.load(path_to_ngame_model, map_location="cpu")
        for k, v in old_state_dict.items():
            name = k.replace("embedding_labels.encoder.transformer.0.auto_model.", "")
            new_state_dict[name] = v
        new_state_dict.keys()
        print(self.model.encoder.transformer.load_state_dict(new_state_dict, strict=True))
        