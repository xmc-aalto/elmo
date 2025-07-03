import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32, 'float8':torch.float8_e4m3fn}


class TransformerEncoder(nn.Module):
    '''
     Custom Transformer Encoder with configurable model components. 
    '''
    def __init__(self, cfg,dtype):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self._dtype = dtype
        self.device = torch.device(cfg.environment.device)
        self.transformer = self.load_transformer_model(cfg)
        self.pooler = self.create_pooler()
        
    def load_transformer_model(self, cfg):
        """ Load transformer model based on the provided configuration. """
        model_config = AutoConfig.from_pretrained(cfg.model.encoder.encoder_model)
        model_config.output_hidden_states = True
        try:
            return AutoModel.from_pretrained(
                cfg.model.encoder.encoder_model, 
                add_pooling_layer=False, 
                config=model_config
            ).to(self.device)
        except Exception as e:
            print(f"Failed to load model with pooling layer removed: {e}")
            return AutoModel.from_pretrained(
                cfg.model.encoder.encoder_model,
                config=model_config
            ).to(self.device)
      
    def forward(self, tokens,masks):
        '''
        Forward pass through transformer and pooling layers. 
        '''
        return self.pooler(self.transformer(tokens,masks),masks).contiguous()
    
    def create_pooler(self):
        '''
         Create a pooling layer based on the configuration.
        '''
        def pool_last_hidden_avg(tf_output, masks):
            last_hidden_state = tf_output['last_hidden_state']
            input_mask_expanded = masks.unsqueeze(-1).expand(last_hidden_state.size()) #.float()
            sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return (sum_hidden_state / sum_mask).to(self._dtype)
        
        def pool_last_nhidden_conlast(tf_output,masks):
            bert_out = tf_output[-1]
            bert_data = [bert_out[-i][:, 0] for i in range(1, self.cfg.model.encoder.feature_layers+1)]
            return torch.cat(bert_data, dim=-1)
        
        def pool_last_hidden_weighted_avg(tf_output, masks):
            last_hidden_state = tf_output['last_hidden_state']  # (batch_size, seq_length, hidden_size)
            batch_size, seq_length, hidden_size = last_hidden_state.size()

            # Create weights that increase towards the end of the sequence
            weights = torch.arange(1, seq_length + 1, dtype=torch.float, device=last_hidden_state.device)
            weights = weights.unsqueeze(0).expand(batch_size, seq_length)  # (batch_size, seq_length)

            # Apply masks to weights
            weights = weights * masks.float()  # Zero out weights where mask is zero

            # Normalize weights so they sum to 1 for each sample
            sum_weights = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            weights = weights / sum_weights  # (batch_size, seq_length)

            # Expand weights to match the hidden size
            weights = weights.unsqueeze(-1)  # (batch_size, seq_length, 1)

            # Compute weighted average
            weighted_avg = torch.sum(last_hidden_state * weights, dim=1)  # (batch_size, hidden_size)
            return weighted_avg.to(self._dtype)
            
        if self.cfg.model.encoder.pool_mode == 'last_hidden_avg':
            return pool_last_hidden_avg
        elif self.cfg.model.encoder.pool_mode == 'last_nhidden_conlast':
            return pool_last_nhidden_conlast
        elif self.cfg.model.encoder.pool_mode == 'last_hidden_weighted_avg':
            return pool_last_hidden_weighted_avg
        else:
            raise ValueError('Invalid pooling mode specified in the configuration.')
            


class SimpleTModel(nn.Module):
    '''
     A simple model with transformer encoder and extreme dense layer fused with loss function.
    '''

    def __init__(self,cfg,path):
        super(SimpleTModel,self).__init__()
        
        self.cfg = cfg
        self.path = path
        self.device = torch.device(cfg.environment.device)
        
        self.encoder_dtype = dtype_map[cfg.model.encoder.dtype]
        self.encoder_dtype = torch.float32 if cfg.training.precision.mode in ['purefp'] else self.encoder_dtype
        self.encoder_dtype = dtype_map[cfg.training.precision.purelp_dtype] if cfg.training.precision.mode in ['purelp'] else self.encoder_dtype

        self.encoder = TransformerEncoder(cfg,self.encoder_dtype).to(self.encoder_dtype)
        self.configure_components(cfg)
        
        # Initialize FP8 Encoder from torchao
        if self.cfg.training.FP8.use_fp8_encoder:
            # optional: filter modules from being eligible for float8 conversion
            from torchao.float8 import convert_to_float8_training
            def module_filter_fn(mod: torch.nn.Module, fqn: str):
                # don't convert the last module
                if fqn == "1":
                    return False
                # don't convert linear modules with weight dimensions not divisible by 16
                if isinstance(mod, torch.nn.Linear):
                    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                        return False
                return True
            convert_to_float8_training(self.encoder, module_filter_fn=module_filter_fn)
            cfg.model.encoder.use_torch_compile = True
        if cfg.model.encoder.use_torch_compile:
            self.encoder = torch.compile(self.encoder)
            
              
    def configure_components(self, cfg):
        """ Configure additional components like dropout, linear layers, etc. based on the model configuration. """
        self.dropout = nn.Dropout(cfg.model.encoder.embed_dropout).to(self.device)

        if cfg.model.bottleneck.use_bottleneck_layer:
            self.bottleneck = nn.Linear(
                cfg.model.encoder.feature_layers * self.encoder.transformer.config.hidden_size,
                cfg.model.bottleneck.bottleneck_size
            ).to(dtype_map[cfg.model.bottleneck.dtype]).to(self.device)

        #select XMC layer type based on implementation and dtype [implementations are in src/xmc_classifiers/]
        if cfg.training.loss_fn=='bce':
            if cfg.model.xmc.implementation=="chunked":
                from xmc_classifiers.chunked import XMCBCECHUNKEDLayer
                self.xfc = XMCBCECHUNKEDLayer(cfg)

            elif cfg.model.xmc.implementation=="fp8chunked":
                from xmc_classifiers.chunked_fp8xmc_clf import Fp8XMCBCEChunkedLayer
                self.xfc = Fp8XMCBCEChunkedLayer(cfg)
            elif cfg.model.xmc.implementation=="fp8chunkedheadkahan":
                from xmc_classifiers.chunked_head_kahan_fp8xmc_clf import Fp8XMCBCEChunkedHeadKahanLayer
                self.xfc = Fp8XMCBCEChunkedHeadKahanLayer(cfg)
            else:
                raise ValueError('Other method than chunked doesn\'t support yet.')
                
        else:
            raise ValueError('Other loss than bce doesn\'t support skip-loss implementation.')
            

    def forward(self,tokens,masks):
        ''' Forward pass through the model. '''
        
        out = self.dropout(self.encoder(tokens,masks))
        
        if self.cfg.model.bottleneck.use_bottleneck_layer:
            out = self.bottleneck(out)
            
        return out
    
    def param_list(self):
        param_list = []

        optimizer_params_encoder = []
        for n, p in self.encoder.named_parameters():
            if p.requires_grad:
                optimizer_params_encoder.append((n, p))
        
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_list += [
                {'params': [p for n, p in optimizer_params_encoder if not any(nd in n for nd in no_decay_params)],
                    'weight_decay': self.cfg.training.encoder.wd, "lr":self.cfg.training.encoder.lr},
                {'params': [p for n, p in optimizer_params_encoder if any(nd in n for nd in no_decay_params)],
                    "lr":self.cfg.training.encoder.lr ,'weight_decay': 0.0}]
            
        if self.cfg.model.bottleneck.use_bottleneck_layer:
            param_list.append({"params":self.bottleneck.parameters(),"lr":self.cfg.training.encoder.lr})
         
        return param_list 