
import torch
from torch import nn
from triton_kernels.stochastic_rounding_kernel import sgd_update 


dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32, 'float8':torch.float8_e4m3fn}

class XMCBCECHUNKEDLayer(nn.Module):
    '''
    Custom XMC layer fused with BCE Loss.
    Vanilla XMC Layer with Chunked updates. Supports FP32, BF16 and FP16
    Doesn't use gradient fusion.
    
    '''

    def __init__(self,cfg):
        super(XMCBCECHUNKEDLayer,self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
        self._dtype = dtype_map[cfg.model.xmc.dtype]
        self._dtype = torch.float32 if cfg.training.precision.mode in ['purefp'] else self._dtype
        self._dtype = dtype_map[cfg.training.precision.purelp_dtype] if cfg.training.precision.mode in ['purelp'] else self._dtype
        
        print(f"self._dtype={self._dtype}")
        
        num_labels = cfg.data.num_labels 
        self.num_labels = num_labels
        self.num_chunks = self.cfg.model.xmc.num_chunks
        self.chunk_size = (num_labels + self.num_chunks - 1) // self.num_chunks 
        

        self.xfc_weight = torch.empty(self.num_labels,cfg.model.xmc.input_features,dtype=self._dtype).to(self.device)
        nn.init.normal_(self.xfc_weight,mean=0.0,std=0.02)
        #nn.init.xavier_uniform_(self.xfc_weight)
        
        #use momentum 
        if cfg.training.xmc.momentum != 0:
            self.momentum_buff = torch.zeros(self.num_labels,cfg.model.xmc.input_features,dtype=self._dtype,device=self.device)
          
        #input gradients
        self.grad_input = torch.zeros(cfg.data.batch_size,cfg.model.xmc.input_features,dtype=self._dtype,device=self.device)
        self.dummy = torch.zeros(1, dtype=self._dtype, device=self.device)
        self.loss = torch.tensor(0.0,device=self.device)
        self.iteration = 0
        print(f" weights dtype={self.xfc_weight.dtype}")
      
    @torch.no_grad()  
    def xfc_forward(self,embed):
        '''
        Mainly used during inference. 
        '''
        return torch.matmul(embed, self.xfc_weight.t()) 
    
    @torch.no_grad() 
    @torch.compile
    def _compute_loss(self, outlogit, rows, cols):
        '''
        Computes the loss for a set of labels.
        '''
        temp = outlogit.clone()
        temp.clamp_(min=0.0)
        loss = temp.sum()
        loss -= outlogit[rows, cols].sum()
        temp.copy_(outlogit)
        temp.abs_()
        temp.neg_()
        temp.exp_()
        temp.add_(1.0)
        temp.log_()
        loss += temp.sum()
        return loss.item()
            
        
    @torch.no_grad() #@torch.compile #increases peak memory consumption
    def xfc_forward_backward(self,embed,labels,xmc_lr,skip_loss=True):
        '''
        Computes activation gradients for the XMC layer fused with BCE Loss.
        Args:
            embed: Encoder or bottleneck layer output.
            labels: Positive labels in sparse format (row,col). shape = (Nb,2) where Nb is number of positive in that batch.
            skip_loss: Whether to skip the loss calculation. 
                        Peak memory friendly to calculate here if using gradient checkpointing.
        '''
        rows,cols = labels[:,0],labels[:,1]
        loss = 0
        bsz = embed.shape[0]
        
        if embed.dtype!=self._dtype:
            embed = embed.to(self._dtype)
        
        #grad input is a small buffer so fp32 accumulation doesn't affect peak memory
        grad_input = torch.zeros(bsz, self.cfg.model.xmc.input_features, dtype=torch.float32, device=self.device)
             
        #chunked classifier update [forward, loss/skip loss, backward, optimizer step]
        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx*self.chunk_size
            end_idx = min(start_idx+self.chunk_size,self.num_labels)
            weight_chunk = self.xfc_weight[start_idx:end_idx,:]
            
            outlogit_chunk = torch.matmul(embed, weight_chunk.t())
            
            mask = (cols>=start_idx) & (cols<end_idx)
            rows_chunk = rows[mask]
            cols_chunk = cols[mask] - start_idx

            # Compute loss for the label chunk
            if not skip_loss:
                loss += self._compute_loss(weight_chunk, rows_chunk, cols_chunk)

            # Loss gradient for the chunked labels
            torch.sigmoid(outlogit_chunk, out=outlogit_chunk)
            outlogit_chunk[rows_chunk, cols_chunk] -= 1.0

            # Compute partial activation gradient contribution from current label chunk
            grad_input += torch.matmul(outlogit_chunk, weight_chunk)
            
            #weight grad calculation
            xfc_weight_grad = outlogit_chunk.t().mm(embed)

            #sgd update
            if self._dtype==torch.float32 or self._dtype==torch.float16: 
                xfc_weight_grad.add_(weight_chunk, alpha=self.cfg.training.xmc.wd)
                weight_chunk.add_(xfc_weight_grad, alpha=-xmc_lr[0])
            else:
                random_int = torch.randint(0, 1000000, (1,)).item()
                sgd_update(weight_chunk,xfc_weight_grad,None,xmc_lr[0],self.cfg.training.xmc.wd,True,random_int)
                          
        #converting fp32 accumulator to original dtype
        self.grad_input = grad_input.to(self._dtype)
        self.iteration += 1

        return loss
