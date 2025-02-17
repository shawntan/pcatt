from pcatt.hf.greedtok import GreedTok
from datasets import load_dataset 
from datasets.arrow_dataset import Dataset
import regex
import math
import torch
if __name__ == "__main__":
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+|[\p{N}]| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    batch_size = 1024
    tokenize: GreedTok = GreedTok()
    # pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # dataset: Dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    test_text = "In 1984, Winston Smith plays the main protagonist..."
    splitted = regex.findall(pattern, test_text)
    print("Pattern splits:", splitted)
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
        batch_iterator(dataset, batch_idxs),
        special_tokens_map={
            "pad_token": "<pad>",
            "unk_token": "<unk>", 
            "eos_token": "<eos>"
        },
        min_word_count=1,
        max_token_size=60,
        vocab_size=2**16,
        pattern=pattern
    )

    tokenize.save_pretrained('./tmp')
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

