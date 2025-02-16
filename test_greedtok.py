from pcatt.hf.greedtok import GreedTok
from datasets import load_dataset 
from datasets.arrow_dataset import Dataset
import regex

if __name__ == "__main__":
    print("Imported.")
    tokenize: GreedTok = GreedTok()
    pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    dataset: Dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    print("Documents:", len(dataset))
    def batch_iterator(dataset, batch_size=512):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]["text"]
            yield batch
    print(len(list(batch_iterator(dataset))))
    tokenize.train_new_from_iterator(
        batch_iterator(dataset),
        special_tokens_map={
            "pad_token":"<pad>",
            "unk_token":"<unk>", 
            "eos_token":"<eos>"
        },
        min_word_count=1,
        max_token_size=1000,
        vocab_size=256,
    )
    original_str = "The quicker brown fox jumps over the lazy dog."

    idxs = tokenize.encode(original_str)
    print(idxs)
    print("Original:", original_str)
    print("Tokens:  ", idxs)
    print("Readable:", [
        tokenize.final_ids_map[x]
        for x in idxs
        if x not in tokenize.special_token_ids
    ])
    print("EncDec:  ", tokenize.decode(idxs))
