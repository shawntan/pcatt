from pcatt.hf.greedtok import GreedTok
from datasets import load_dataset 
from datasets.arrow_dataset import Dataset
import regex
import math
import torch
if __name__ == "__main__":
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+|[\p{N}]| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    batch_size = 1024
    min_word_count = 1
    vocab_size = 2**16
    max_token_size = 20

    out_file = f"pcatt_vocab_size_{vocab_size}-min_word_count_{min_word_count}-max_token_size_{max_token_size}"
    print(out_file)
    tokenize: GreedTok = GreedTok()
    # pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    test_text = "In 1984, Winston Smith is the main protagonist..."
    splitted = regex.findall(pattern, test_text)
    print("Pattern splits:", splitted)
    # dataset: Dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    dataset: Dataset = load_dataset("Zyphra/dclm-dedup", split="train", num_proc=32)
    num_docs = len(dataset)
    print("Documents:", num_docs)
    num_batches = int(math.ceil(num_docs / batch_size))
    batch_idxs = torch.randperm(num_batches)
    batch_idxs = batch_idxs[:num_batches // 5]
    batch_idxs, _ = batch_idxs.sort()
    print("Keeping %d batches." % batch_idxs.size(0))

    def batch_iterator(dataset, batch_idxs, batch_size=512):
        for i in batch_idxs:
            batch = dataset[i: i + batch_size]["text"]
            yield batch

    tokenize = tokenize.train_new_from_iterator(
        batch_iterator(dataset, batch_idxs, batch_size=batch_size),
        special_tokens_map={
            "pad_token": "<pad>",
            "unk_token": "<unk>", 
            "eos_token": "<eos>"
        },
        min_word_count=min_word_count,
        # max_token_size=60,
        max_token_size=max_token_size,
        vocab_size=vocab_size - 256,
        pattern=pattern
    )

    tokenize.save_pretrained(out_file)
    print(len(tokenize))
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
    print("eos_token_id:", tokenize.eos_token_id)
    print("pad_token_id:", tokenize.pad_token_id)
    print([tokenize.final_ids_map[x] for x in [0, 2]])
 

