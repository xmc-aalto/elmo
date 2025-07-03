import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import numpy as np
import os
from preprocess import tokenize_file



name_map = { 'amazon670k': 'Amazon-670K', 'amazontitles670k':'AmazonTitles-670K','wiki500k': 'Wiki-500K','amazon3m':'Amazon-3M', 
             'lfamazontitles131k':'LF-AmazonTitles-131K', 'lfwikiseealso320k':'LF-WikiSeeAlso-320k', 'lfamazontitles1.3m':'LF-AmazonTitles-1.3M'}
dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16}


def collate(batch):
    '''
    collate function to be used when sparse label format is needed.
    
    '''
    tokens = []
    attention_mask = []
    labels = []
    for i, (t, m, l) in enumerate(batch):
        tokens.append(t)
        attention_mask.append(m)
        l_coo = [(i, lbl) for lbl in l]
        labels += l_coo
    return (
        torch.utils.data.default_collate(tokens),
        torch.utils.data.default_collate(attention_mask),
        torch.Tensor(labels).to(torch.int32).contiguous(),
    )
    


class DataHandler:
    '''
    Handle all the data reading, preprocessing ,dataset, dataloader and other stuff.
    
    '''
    def __init__(self,cfg,path):
        self.cfg = cfg
        self.path = path
        self.device = torch.device(cfg.environment.device)
        if cfg.training.precision.mode=="purelp":
            self.low_precision_dtype = dtype_map[cfg.training.precision.purelp_dtype] 
        
        self.label_map = {}
        if cfg.data.tokenized_loading:
            self.read_tokenized_files()
        else:
            self.read_files()
        

    def read_tokenized_files(self):
        
        if not self.cfg.data.is_lf_data: 
            
            train_raw_texts = self._read_text_files(self.path.train_raw_texts) 
            test_raw_texts = self._read_text_files(self.path.test_raw_texts) 

            self.train_labels = self._read_label_files(self.path.train_labels)
            self.test_labels = self._read_label_files(self.path.test_labels)
            train_filename = str(self.cfg.data.dataset)+'_'+'train'
            test_filename = str(self.cfg.data.dataset)+'_'+'test'
        else:
            train_raw_texts, train_labels = self._read_lf_files(self.path.train_json)
            self.train_labels = train_labels
            test_raw_texts, self.test_labels = self._read_lf_files(self.path.test_json)
            train_filename = str(self.cfg.data.dataset)+'_'+'train'
            test_filename = str(self.cfg.data.dataset)+'_'+'test'
            if self.cfg.data.augment_label_data:
                label_raw_texts,label_labels = self._read_lf_files(self.path.label_json,label_json=True)
                train_raw_texts += label_raw_texts
                self.train_labels += label_labels
                train_filename = str(self.cfg.data.dataset)+'_'+'train_'+'aug'
                test_filename = str(self.cfg.data.dataset)+'_'+'test'

                
        if not self.cfg.data.label_sort:
            for i, k in enumerate(sorted(self.label_map.keys())):
                self.label_map[k] = i
        else:
            sorted_labels = sorted(self.label_map.items(), key=lambda x: x[1], reverse=True)
            # Reset label_map and assign new indices
            self.label_map = {}
            for idx, (label, freq) in enumerate(sorted_labels):
                self.label_map[label] = idx
                
        self.train_nsample = len(train_raw_texts)
        self.test_nsample = len(test_raw_texts)
                
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.encoder.encoder_tokenizer,do_lower_case=True)
        self.pad_token_id = tokenizer.pad_token_id
        self.train_tokenized_filename = train_filename+'_'+str(self.cfg.data.max_len)+'.dat'
        if not os.path.exists(self.train_tokenized_filename):
            print(f"tokenized train file is not available. creating the file")
            tokenize_file(train_raw_texts,self.train_tokenized_filename,tokenizer,self.train_nsample,self.cfg.data.max_len)
        else:
            print(f"tokenized train file exists. File {self.train_tokenized_filename } would be used.")
        
        self.test_tokenized_filename = test_filename+'_'+str(self.cfg.data.max_len)+'.dat'
        if not os.path.exists(self.test_tokenized_filename):
            print(f"tokenized test file is not available. creating the file")
            tokenize_file(test_raw_texts,self.test_tokenized_filename,tokenizer,self.test_nsample,self.cfg.data.max_len)
            
        else:
            print(f"tokenized test file exists. File {self.test_tokenized_filename } would be used")
        

    def read_files(self):
        
        if not self.cfg.data.is_lf_data:
            self.train_raw_texts = self._read_text_files(self.path.train_raw_texts) 
            self.test_raw_texts = self._read_text_files(self.path.test_raw_texts) 

            self.train_labels = self._read_label_files(self.path.train_labels)
            self.test_labels = self._read_label_files(self.path.test_labels)
        else:
            self.train_raw_texts, train_labels = self._read_lf_files(self.path.train_json)
            self.train_labels = train_labels
            self.test_raw_texts, self.test_labels = self._read_lf_files(self.path.test_json)
            if self.cfg.data.augment_label_data:
                label_raw_texts,label_labels = self._read_lf_files(self.path.label_json,label_json=True)
                self.train_raw_texts += label_raw_texts
                self.train_labels += label_labels

        if not self.cfg.data.label_sort:
            for i, k in enumerate(sorted(self.label_map.keys())):
                self.label_map[k] = i
        else:
            sorted_labels = sorted(self.label_map.items(), key=lambda x: x[1], reverse=True)
            # Reset label_map and assign new indices
            self.label_map = {}
            for idx, (label, freq) in enumerate(sorted_labels):
                self.label_map[label] = idx
                       
        
    def _read_text_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            container.append(line.strip())
    
        return container

    def _read_label_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            for l in line.strip().split():
                self.label_map[l] = self.label_map.get(l,0)+1 # count frequency
            container.append(line.strip().split())
            
        return container
    
    def _read_lf_files(self,file,label_json=False):
        text_data = []
        labels = []
        key = 'title' if 'titles' in self.cfg.data.dataset else 'content'
        if label_json:
            key='title'
        with open(file) as f:
            for i,line in enumerate(f):
                data = json.loads(line)
                if 'titles' in self.cfg.data.dataset or label_json:
                    text_data.append(data["title"])
                else:
                    text_data.append(data["title"] + " " + data["content"])
                if label_json:
                    labels.append([i]) # no need to count this frquency. add 1 to all labels.
                    self.label_map[i] = self.label_map.get(i, 0) + 1 
                else:
                    lbls = data['target_ind']
                    for l in lbls:
                        self.label_map[l]= self.label_map.get(l,0)+1 # count frquency
                    labels.append(lbls)

        return text_data,labels
        
    
    def getDatasets(self):
        
        if self.cfg.data.tokenized_loading:
            train_dset = SimpleTokenizedDataset(self.cfg,self.train_tokenized_filename,self.train_nsample,
                                                self.train_labels,self.label_map,self.pad_token_id,mode='train',task='train')
            test_dset = SimpleTokenizedDataset(self.cfg,self.test_tokenized_filename,self.test_nsample,
                                               self.test_labels,self.label_map,self.pad_token_id,mode='test')
            train_dset_eval = SimpleTokenizedDataset(self.cfg,self.train_tokenized_filename,self.train_nsample,
                                                     self.train_labels,self.label_map,self.pad_token_id,mode='train')
        else:
            train_dset = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,mode='train',task='train')
            test_dset = SimpleDataset(self.cfg,self.test_raw_texts,self.test_labels,self.label_map,mode='test')
            train_dset_eval = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,mode='train')
            
        return train_dset,test_dset,train_dset_eval
    

    def getDataLoader(self,dset,mode='train'):
        '''
        #currently  separate dataloader for train (sparse labels) and evaluate (dense labels) in order to fasten both process.
        
        '''
        assert mode in ['train','test'], " mode must be either train or test."
        shuffle=False
        batch_size = self.cfg.data.test_batch_size
        workers = self.cfg.data.num_workers
        pin_mem = True
        drop_last = False
        if mode == 'train':
            shuffle=True
            batch_size = self.cfg.data.batch_size
            workers = self.cfg.data.num_workers
            pin_mem=True
            if self.cfg.model.xmc.implementation in ["fp8chunked", "fp8chunkedheadkahan"]:
                drop_last = True
        return DataLoader(dset, batch_size=batch_size, num_workers=workers, pin_memory=pin_mem, persistent_workers = True, prefetch_factor=2,
                          drop_last=drop_last,shuffle=shuffle, collate_fn=collate)


class SimpleDataset(Dataset):
    '''
    Tokenization on the fly during batching.
    
    '''
    
    def __init__(self,cfg,raw_texts,labels,label_map,mode='train',task='evaluate'):
        super(SimpleDataset).__init__()
        
        self.cfg = cfg
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder.encoder_tokenizer,do_lower_case=True)
        self.cls_token_id = [101]  # [self.tokenizer.cls_token_id]
        self.sep_token_id = [102]  # [self.tokenizer.sep_token_id]
        self.raw_text = raw_texts
        self.labels = labels
        self.label_map = label_map
        self.mode = mode
                               
    def __len__(self):
        return len(self.raw_text)
    
    def __getitem__(self,idx):
        
        padding_length = 0
        raw_text = self.raw_text[idx]
        tokens = self.tokenizer.encode(raw_text, add_special_tokens=False,truncation=True, max_length=self.cfg.data.max_len)
        tokens = tokens[:self.cfg.data.max_len-2]
        tokens = self.cls_token_id +tokens + self.sep_token_id
        
        if len(tokens)<self.cfg.data.max_len:
            padding_length = self.cfg.data.max_len - len(tokens)
        attention_mask = torch.tensor([1] * len(tokens) + [0] * padding_length)
        tokens = torch.tensor(tokens+([0]*padding_length))
        
        labels = [self.label_map[i] for i in self.labels[idx] if i in self.label_map]

        return tokens, attention_mask, labels
 
    
class SimpleTokenizedDataset(Dataset):
    '''
    Requires pre-tokenized data.
    
    '''
    
    def __init__(self,cfg,tokenized_filename,nsample,labels,label_map,pad_token_id,mode='train',task='evaluate'):
        super(SimpleTokenizedDataset,self).__init__()
        
        self.cfg = cfg
        self.task = task
        self.mode = mode
        self.labels = labels
        self.label_map = label_map
        self.pad_token_id = pad_token_id 
        self.num_samples = nsample
        self.tokenized_filename = tokenized_filename
        
        self.data_shape = (self.num_samples, self.cfg.data.max_len)
        self._memmap_initialized = False

    def _initialize_memmap(self):
        """Lazy initialization of the tokenized memmap array.
        Also helps to avoid multiple copies over num_workers which is the case during loading in the constructor.
        
        """
        if not self._memmap_initialized:
            self.memmap_array = np.memmap(self.tokenized_filename, dtype='int32', mode='r', shape=self.data_shape)
            self._memmap_initialized = True
                               
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        if not self._memmap_initialized:
            self._initialize_memmap()    
        
        tokens = torch.tensor(self.memmap_array[idx])
        attention_mask = (tokens != self.pad_token_id).long()
        labels = [self.label_map[i] for i in self.labels[idx] if i in self.label_map]
        return tokens, attention_mask, labels