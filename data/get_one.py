import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def get_last_shard_index(data_cache_dir):
    """ Determine the last processed shard index based on existing files. """
    shard_files = [f for f in os.listdir(data_cache_dir) if f.endswith('.bin')]
    if not shard_files:
        return -1  # No shard files found
    shard_indices = [int(f.split('_')[-1].split('.')[0]) for f in shard_files]
    return max(shard_indices)

# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
if args.version == "10B":
    local_dir = "fineweb10B"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "fineweb100B"
    remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Determine the last processed shard index
last_shard_index = get_last_shard_index(DATA_CACHE_DIR)

# download the dataset metadata and stream the data
fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train", streaming=True)

# Take only a small subset of the dataset
small_subset = [next(iter(fw)) for _ in range(100)]  # Adjust the number of samples as needed

# Transform small_subset to a dictionary format expected by Dataset.from_dict
columns = small_subset[0].keys()
small_subset_dict = {col: [item[col] for item in small_subset] for col in columns}

fw = Dataset.from_dict(small_subset_dict)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special token that delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2)  # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = last_shard_index + 1  # Start from the next shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    doc_index = 0

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # Skip already processed documents if needed
        if doc_index < shard_index:
            doc_index += 1
            continue
        
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
