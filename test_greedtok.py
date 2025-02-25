from pcatt.hf.greedtok import GreedTok
from datasets import load_dataset 
from datasets.arrow_dataset import Dataset
import regex
import math
import torch
import sys
if __name__ == "__main__":
    tokenize = GreedTok.from_pretrained(sys.argv[1])
    original_str = "The quick brown fox jumps over the lazy dog.\n"
    # idxs = tokenize(original_str)
    idxs = tokenize([original_str])['input_ids'][0]
    print(idxs)
    print("Original:", original_str)
    print("Tokens:  ", idxs)
    print("Readable:", [
        tokenize.final_ids_map[x]
        for x in idxs
        if x not in tokenize.special_token_ids
    ])
    print("EncDec:  ", tokenize.decode(idxs))

