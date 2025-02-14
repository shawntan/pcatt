from pcatt.hf.greedtok import GreedTok
from datasets import load_dataset 
from datasets.arrow_dataset import Dataset
import regex

if __name__ == "__main__":
    print("Imported.")
    tokenize = GreedTok()
    pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    dataset: Dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    print("Documents:", len(dataset))
    def batch_iterator(dataset, batch_size=512):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]["text"]
            words = [
                [regex.sub(" ", "Ġ", w) for w in regex.findall(pat_str, batch[j])] 
                for j in range(len(batch))
            ]

            yield words
    print(len(list(batch_iterator(dataset))))
    tokenize.train_new_from_iterator(batch_iterator(dataset), vocab_size=512)
