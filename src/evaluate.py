import numpy as np
import torch
import scipy.sparse as sp
import xclib.evaluation.xc_metrics as xc_metrics
import itertools
from utils import timeit,dense_to_sparse
from tqdm import tqdm
from scipy.sparse import csr_matrix


def sparsification(labels,label_map,num_labels):
    """
    Convert labels into a sparse CSR (Compressed Sparse Row) matrix.

    This function takes a list of label lists (where each inner list contains the labels for a single sample),
    maps these labels to a new label space if necessary, and returns a CSR matrix indicating the presence
    or absence of labels for each sample.

    Parameters:
    - labels (list of lists of int or string): The labels for each sample. Each inner list contains the labels assigned to that sample.
    - label_map (dict): A mapping from original label IDs to new label IDs. Used to convert labels to a new label space.
    - num_labels (int): The total number of unique labels in the new label space.

    Returns:
    - csr_matrix: A CSR matrix of shape (len(labels), num_labels), where each row represents a sample and each column represents a label.
                  The entries in the matrix are 1 if the label is present for the sample, and 0 otherwise.
    """
    
    labels_tmp = [[label_map[y] for y in x] for x in labels]
    rows = list(itertools.chain(*[[x]*len(y) for x,y in zip(range(len(labels_tmp)),labels_tmp)]))
    val = np.ones(len(rows))
    y_true = sp.csr_matrix((val, (rows, list(itertools.chain(*labels_tmp)))), shape=(len(labels_tmp),num_labels))
    return y_true
    

class Evaluator:
    """
    A class for evaluating the performance of models on a given dataset using various metrics.

    This class supports two modes of operation ('memory_efficient' and 'debug') to accommodate datasets of different sizes.
    It calculates various evaluation metrics, including precision at K, recall at K, and others, depending on the configuration.
    For larger datasets, the 'memory_efficient' mode is recommended.

    Parameters:
    - topk (int): The number of top predictions to consider when calculating metrics.
    - cfg: A configuration object containing model and evaluation settings, such as device, number of labels, and metric flags.
    - mode (str): The mode of operation, which can be either 'memory_efficient' or 'debug'. Influences how data is processed.
    - eval_recall (bool): Flag indicating whether to calculate recall metrics.
    - eval_psp (bool): Flag indicating whether to calculate propensity-scored precision metrics.
    - eval_psr (bool): Flag indicating whether to calculate propensity-scored recall metrics.
    - eval_ndcg (bool): Flag indicating whether to calculate NDCG metrics.
    - label_map (dict): A mapping from original label IDs to new label IDs, used for converting labels to a consistent label space.
    - y_true (csr_matrix): The ground truth labels in sparse CSR format, used for comparison with predictions.

    Methods:
    - Calculate_Metrics(data_loader, model): Calculates the configured metrics for the given model and dataset.
    - _Calculate_Metrics_DEBUG(data_loader, model): Debug mode metric calculation, supporting various metrics.
    - _Calculate_Metrics_ME(data_loader, model): Memory-efficient metric calculation, currently supports precision at K.

    Note:
    Currently 'memory_efficient' mode only supports P@K metrics.
    """
    
    def __init__(self,cfg,label_map,true_labels,mode='memory_efficient',train_labels = None,topk=100,filter_path=None):
        """
        Initializes the Evaluator with the necessary configuration, label mappings, and mode of operation.

        Args:
        - cfg: Configuration object containing settings such as the device, number of labels, and which metrics to evaluate.
        - label_map (dict): Mapping from original label IDs to new label IDs for consistent label representation.
        - true_labels (list of lists of int): The true labels for each sample in the dataset.
        - mode (str, optional): The mode of operation ('memory_efficient' or 'debug'). Defaults to 'memory_efficient'.
        - train_labels (list of lists of int/str, optional): The labels for each sample in the training set, required for some metrics.
        - topk (int, optional): The number of top predictions to consider for metric calculations. Defaults to 10.

        Raises:
        - ValueError: If `train_labels` is not provided when required for propensity-scored metric calculations in 'debug' mode.
        """

        self.topk = topk
        self.cfg = cfg
        self.mode = mode
        self.eval_recall = cfg.training.evaluation.eval_recall
        self.eval_psp = cfg.training.evaluation.eval_psp
        self.eval_psr = cfg.training.evaluation.eval_psr
        self.eval_ndcg = cfg.training.evaluation.eval_ndcg
        self.filter_mat = None
        self.device = torch.device(cfg.environment.device)
        
        
        assert self.mode in ['memory_efficient','debug'], f"mode can be either large or small not {mode}"
        if self.eval_psp and train_labels is None and self.mode=='debug':
            raise ValueError("train_labels should be passed in order to calculate propensity based metrics")
        self.label_map = label_map
        if self.mode == 'debug':
            self.y_true = sparsification(true_labels,label_map,cfg.data.num_labels)
            if filter_path is not None:
                temp = np.fromfile(filter_path, sep=' ').astype(int)
                temp = temp.reshape(-1, 2).T
                # Map the column indices using self.label_map
                mapped_cols = np.array([self.label_map[col] for col in temp[1]])
                # Create the sparse matrix
                self.filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], mapped_cols)),shape=self.y_true.shape).tocsr()
                
        if self.eval_psp and self.mode=='debug':
            y_train = sparsification(train_labels,label_map,cfg.data.num_labels)
            num_instances, _ = y_train.shape
            freqs = np.ravel(np.sum(y_train, axis=0))
            C = (np.log(num_instances)-1)*np.power(cfg.training.evaluation.B+1, cfg.training.evaluation.A)
            wts = 1.0 + C*np.power(freqs+cfg.training.evaluation.B, -cfg.training.evaluation.A)
            self.wts = np.ravel(wts)
        
    def _prediction_matrix_dense(self,data_loader,model):
        """
        Generates a dense prediction matrix from model predictions for a dataset.

        Args:
        - data_loader: DataLoader providing batches of data (tokens, masks, labels).
        - model:  Model for generating predictions.

        Returns:
        - A dense numpy array of predictions with shape (N, num_labels), where N is the dataset size.
        """
        model.eval()
        pred_matrix = torch.zeros((1,self.cfg.data.num_labels))
        with torch.no_grad():
            for step,data in tqdm(enumerate(data_loader)):
                tokens,mask,labels,_ = data
                tokens, mask, labels = tokens.to(self.device),mask.to(self.device),labels.to(self.device)
                logits,_ = model(tokens,mask)
                pred_matrix = torch.cat((pred_matrix,logits.cpu()),dim=0)
        
        pred_matrix = pred_matrix[1:].numpy()
        
        return pred_matrix
    
    def _prediction_matrix_sparse(self,data_loader,model):
        """
        Builds a sparse prediction matrix applicable for large datasets.

        Parameters:
        - data_loader: Yields batches of data.
        - model: Model to evaluate.

        Returns:
        - Sparse prediction matrix as csr_matrix.
        """
        model.eval()
        rows, cols, vals = torch.zeros([1]).to(self.device), torch.zeros([1]).to(self.device), torch.zeros([1]).to(self.device)
        with torch.no_grad():
            for step,data in tqdm(enumerate(data_loader)): #,total=len(data_loader
                tokens,mask,labels = data
                bsz = tokens.shape[0]
                tokens, mask, labels = tokens.to(self.device),mask.to(self.device,dtype=torch.bfloat16),labels.to(self.device)
                embed = model(tokens,mask)
                if embed.dtype!=model.xfc._dtype:
                    embed = embed.to(model.xfc._dtype)

                outlogit = model.xfc.xfc_forward(embed)
                val,ind = torch.topk(outlogit,k=self.topk,sorted=False)
                #print(f"predicted val shape:{val.shape}")
                cols = torch.cat([cols,ind.flatten()])
                start = step*data_loader.batch_size
                rows = torch.cat([rows,torch.arange(start,start+bsz).repeat_interleave(self.topk).flatten().to(self.device)])
                vals = torch.cat([vals,val.flatten()])
                
        
        rows, cols, vals = rows[1:].cpu().numpy(), cols[1:].cpu().numpy(), vals[1:].cpu().numpy()
        pred_matrix = csr_matrix((vals, (rows, cols)), shape=self.y_true.shape)
        
        return pred_matrix

    
    def _Calculate_Metrics_DEBUG(self,data_loader,model):
        """
        Computes evaluation metrics using Xclib library. More suitable for debug and developement purposes.

        Args:
        - data_loader: Provides batches of data.
        - model: The model being evaluated.

        Returns:
        - A dictionary of calculated metrics (e.g., 'P@K', 'R@K') for dataset.
        """

        pred_matrix = self._prediction_matrix_sparse(data_loader,model)
        #print(f"prediction matrix shape:{pred_matrix.shape}")
        
        #filtering part
        if self.filter_mat is not None:
            temp = self.filter_mat.tocoo()
            pred_matrix[temp.row, temp.col] = 0
            pred_matrix = pred_matrix.tocsr()
            pred_matrix.eliminate_zeros()
        
        pk = xc_metrics.precision(pred_matrix, self.y_true, k=5, sorted=False, use_cython=False).tolist()
        metrics = {'P@K':pk}
        if self.eval_psp:
            metrics['PSP@K'] = xc_metrics.psprecision(pred_matrix, self.y_true, self.wts, k=5, sorted=False, use_cython=False).tolist()
        if self.eval_recall:
            metrics['R@K'] = xc_metrics.recall(pred_matrix, self.y_true, k=5, sorted=True, use_cython=False).tolist()
        if self.eval_psr:
            metrics['PSR@K'] = xc_metrics.psrecall(pred_matrix, self.y_true, self.wts, k=5, sorted=True, use_cython=False).tolist()
        if self.eval_ndcg:
            metrics['NDCG@K'] = xc_metrics.ndcg(pred_matrix, self.y_true, k=5, sorted=True, use_cython=False).tolist()
            
        return metrics
    
    def _Calculate_Metrics_ME(self,data_loader,model):
        """
        Calculates precision at K for large datasets in a memory-efficient and faster manner.

        Parameters:
        - data_loader: DataLoader yielding batches of data and labels. expects the ground truth labels of dataloader in sparse format.
                                                                shape :(Np,2) where Np is the number of positive labels for that batch. 
        - model: PyTorch model for computing predictions.

        Returns:
        - Dictionary with 'P@K' key, containing precision at K (K=1,2,3,4,5) as a list.
        
        Note:
            Currently supports P@K only.
        """

        model.eval()
        with torch.no_grad():
            Pk = [0, 0, 0,0,0]
            for data in tqdm(data_loader):
                tokens,mask,labels = data
                set2 = {tuple(sorted(pair)) for pair in labels.tolist()}
                bsz = tokens.shape[0]
                tokens, mask = tokens.to(self.device),mask.to(self.device,dtype=torch.bfloat16)
                embed = model(tokens,mask)
                if embed.dtype!=model.xfc._dtype:
                    embed = embed.to(model.xfc._dtype)
                outlogit = model.xfc.xfc_forward(embed)
                _,topk_indices = outlogit.topk(5)
                for k in [1,2,3,4,5]:
                    topk_sparse = torch.cat([torch.arange(0,bsz).repeat_interleave(k).view(-1,1).to(self.device),topk_indices[:,:k].reshape(-1,1)],dim=-1)
                    #print(topk_sparse)
                    set1 = {tuple(sorted(pair)) for pair in topk_sparse.tolist()}
                    Pk[k-1] += sum(1 for pair in set2 if pair in set1)*1/(k*bsz)

            Pk = [x/len(data_loader) for x in Pk]
            
            metrics = {'P@K':Pk}

        return metrics
    
    #@timeit
    def Calculate_Metrics(self,data_loader,model):
        if self.mode =='memory_efficient':
            return self._Calculate_Metrics_ME(data_loader,model)
        elif self.mode =='debug':
            return self._Calculate_Metrics_DEBUG(data_loader,model)
        



