import torch.multiprocessing as mp
from transformers import AutoTokenizer
import numpy as np
import time
import functools
from tqdm import tqdm
import math

def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer

def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])

def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    #attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids #, attention_mask

@timeit
def tokenize_dump_memmap(corpus, tokenizer, max_len, filename, num_threads, batch_size=10000000):
    ii = np.memmap(filename, dtype='int64', mode='w+', shape=(len(corpus), max_len))
    for i in tqdm(range(0, len(corpus), batch_size)):
        _input_ids = convert(corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ii[i: i + _input_ids.shape[0], :] = _input_ids



# trnX = [x.strip() for x in open(f'{DATA_DIR}/train_raw_texts.txt', "r", encoding="utf-8").readlines()]
# tstX = [x.strip() for x in open(f'{DATA_DIR}/test_raw_texts.txt', "r", encoding="utf-8").readlines()]

# if args.tokenize_label_texts:
#     Y = [x.strip() for x in open(f'{DATA_DIR}/Y.txt', "r", encoding="latin-1").readlines()]
#     print(len(Y))
#     print(Y[:10])

# print(len(trnX), len(tstX))
# print(trnX[:10], tstX[:10])

# max_len = args.max_length
# tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type, do_lower_case=True)


# print(f"Dumping files in {tokenization_dir}...")
# print("Dumping for trnX...")
# tokenize_dump_memmap(trnX, tokenization_dir, tokenizer, max_len, "trn_doc", args.num_threads)
# print("Dumping for tstX...")
# tokenize_dump_memmap(tstX, tokenization_dir, tokenizer, max_len, "tst_doc", args.num_threads)

# if args.tokenize_label_texts:
#     print("Dumping for Y...")
#     tokenize_dump_memmap(Y, tokenization_dir, tokenizer, max_len, "lbl", args.num_threads)



def tokenize_file(raw_texts,filename,tokenizer,num_samples,max_len):
    data_shape = (num_samples, max_len)
    
    memmap_array = np.memmap(filename, dtype='int32', mode='w+', shape=data_shape)
    # Define the batch size for tokenization
    batch_size = 10000  # Adjust this based on your system's memory capacity

    num_batches = math.ceil(num_samples / batch_size)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_texts = raw_texts[start_idx:end_idx]

        # Tokenize the batch
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=False,
            return_tensors=None)

        # Write the batch tokens to the memmap array
        memmap_array[start_idx:end_idx] = encodings['input_ids']
    
    memmap_array.flush()
    del memmap_array