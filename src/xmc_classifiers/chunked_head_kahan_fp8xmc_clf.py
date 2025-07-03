
import torch
from torch import nn
from triton_kernels.matmul_kernel import large_k_matmul
from triton_kernels.fuse_kernel import matmul_update, fakefp8_matmul_update, kahan_matmul_update



dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32, 'float8':torch.float8_e4m3fn}

class Fp8XMCBCEChunkedHeadKahanLayer(nn.Module):
    '''
    Chunked FP8 XMC classifier
    '''

    def __init__(self, cfg):
        super(Fp8XMCBCEChunkedHeadKahanLayer, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
       
        self._dtype = torch.float8_e4m3fn
        
        print(f"self._dtype={self._dtype}")
        self.fake_fp8_matmul = cfg.training.xmc.simulated_fp8
        self.using_head_kahan = cfg.training.FP8.using_head_kahan
        self.num_chunks = self.cfg.model.xmc.num_chunks
        self.num_labels = cfg.data.num_labels + 16*self.num_chunks - cfg.data.num_labels % (16*self.num_chunks)
        self.num_labels_i = self.num_labels // self.num_chunks
        assert self.num_labels_i % 16 == 0
        xfc_weight = torch.empty(self.num_labels, cfg.model.xmc.input_features, dtype=torch.float32)
        nn.init.xavier_uniform_(xfc_weight)

        self.xfc_weight = nn.ParameterList([nn.Parameter(xfc_weight[i*self.num_labels_i : (i+1)*self.num_labels_i].to(torch.float8_e4m3fn).to(self.device)) for i in range(self.num_chunks)])
        n_labels = 0
        for i in range(self.num_chunks):
            n_labels += self.xfc_weight[i].shape[0]
            self.xfc_weight[i].requires_grad = False
            assert self.xfc_weight[i].shape[0] == self.num_labels_i
        assert n_labels == self.num_labels

        if self.using_head_kahan:
            self.fp8_head_kahan = cfg.training.FP8.fp8_head_kahan
            self.num_head = int(cfg.training.FP8.head_ratio * self.num_labels)
            if self.fp8_head_kahan:
                self.head_kahan = torch.zeros(self.num_head, cfg.model.xmc.input_features, dtype=torch.float8_e5m2, device=self.device)
            else:
                self.head_kahan = torch.zeros(self.num_head, cfg.model.xmc.input_features, dtype=torch.bfloat16, device=self.device)
            assert self.num_head <= self.num_labels_i
        self.block_size = 16
        self.loss = torch.tensor(0.0,device=self.device)
        self.iteration = 0
        print(f" weights dtype={self.xfc_weight[0].dtype}")
      
    @torch.no_grad()  
    def xfc_forward(self, embed):
        '''
        Mainly used during inference
        '''
        for i in range(self.num_chunks):
            if self.fake_fp8_matmul:
                outlogit_i = torch.matmul(embed.to(torch.bfloat16), self.xfc_weight[i].t().to(torch.bfloat16))
            else:
                outlogit_i = torch._scaled_mm(embed.to(torch.float8_e4m3fn), self.xfc_weight[i].t(), out_dtype=torch.bfloat16,
                    scale_a=torch.tensor(1.0).to(self.device), scale_b=torch.tensor(1.0).to(self.device))
            if i == 0:
                outlogit = outlogit_i
            else:
                outlogit = torch.cat((outlogit, outlogit_i), 1)
        return outlogit[:, :self.cfg.data.num_labels]
    
    @torch.no_grad() 
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
    def xfc_forward_backward(self, embed, labels, xmc_lr, skip_loss=True):
        '''
        Computes activation gradients for the XMC layer fused with BCE Loss.
        Args:
            embed: Encoder or bottleneck layer output.
            labels: Positive labels in sparse format.
            skip_loss: Whether to skip the loss calculation. 
                        Peak memory friendly to calculate here if using gradient checkpointing.
        '''
        rows, cols = labels[:,0], labels[:,1]
        loss = 0

        grad_input = torch.zeros_like(embed, dtype=torch.float32, device=self.device)
        #chunking
        for i in range(self.num_chunks):
            if self.fake_fp8_matmul:
                bf16_weight_i = self.xfc_weight[i].to(torch.bfloat16)
                outlogit_i = torch.matmul(bf16_weight_i, embed.to(torch.bfloat16).t())          
            else:
                outlogit_i = torch._scaled_mm(self.xfc_weight[i], embed.to(torch.float8_e4m3fn).t(), out_dtype=torch.bfloat16,
                                    scale_a=torch.tensor(1.0).to(self.device), scale_b=torch.tensor(1.0).to(self.device))
            

            rows_i, cols_i = self.filter_non_local_inds(rows, cols, i*self.num_labels_i, (i+1)*self.num_labels_i)

            outlogit_i.sigmoid_()
            outlogit_i[cols_i, rows_i] -= 1.0 #(num_labels, batch_size)
            if self.fake_fp8_matmul:
                grad_input += large_k_matmul(outlogit_i, bf16_weight_i, bs=self.block_size)
            else:    
                grad_input += large_k_matmul(outlogit_i, self.xfc_weight[i], bs=self.block_size)
            seed = torch.randint(0, 1000000, (1,)).item()
            if self.fake_fp8_matmul:
                fakefp8_matmul_update(outlogit_i, embed, bf16_weight_i, xmc_lr[0], seed, bs=self.block_size)
                self.xfc_weight[i].copy_(bf16_weight_i.to(torch.float8_e4m3fn))
            else:
                if self.using_head_kahan and i == 0:
                    # Kahan head update
                    kahan_matmul_update(outlogit_i[:self.num_head], embed, self.xfc_weight[i][:self.num_head], self.head_kahan, xmc_lr[0], seed, bs=self.block_size)
                    
                    matmul_update(outlogit_i[self.num_head:], embed, self.xfc_weight[i][self.num_head:], xmc_lr[0], seed, bs=self.block_size)
                else:
                    matmul_update(outlogit_i, embed, self.xfc_weight[i], xmc_lr[0], seed, bs=self.block_size)

        self.iteration += 1
        self.grad_input = grad_input.to(torch.bfloat16)
        return loss
    def filter_non_local_inds(self, rows, cols, start, end):
        mask = (cols >= start) & (cols < end)
        return rows[mask], cols[mask] - start