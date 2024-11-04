
import os
import pickle
from tokenizers import SentencePieceUnigramTokenizer

class word_count_iterator:
    def __init__(self, wc):
        self.wc = list(wc.items())
        self.size = sum(wc.values())
        self.item_count = 0
        self.word_count = 0
        
    def __iter__(self):
        self.item_count = 0
        self.word_count = 0
        return self
    
    def __next__(self):
        if self.word_count > self.wc[self.item_count][1]:
            self.item_count += 1
            self.word_count = 0
        
        if self.item_count >= len(self.wc):
            raise StopIteration
            
        self.word_count += 1
        return self.wc[self.item_count][0]
    
"""
# Quick execute
nohup python huggingface_sp_unigram.py --domain 'un' > logs/unigram.log &
nohup python huggingface_sp_unigram.py --domain 'arxiv' > logs/unigram2.log &

"""
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--domain", dest="domain")
args = parser.parse_args()
    
search = {
    'un': [2640,1344,794,516,350,242,175,129],
    'arxiv': [3630,1864,1105,708,478,333,237,171],
}
vocab_sizes = search[args.domain]
vocab_sizes.extend(list(range(500,5000,500)))
    
for vocab_size in vocab_sizes:
    out_dir = f'python_outputs/hf_sp_unigram/{args.domain}_model_{vocab_size}/'
    if os.path.exists(f"{out_dir}/unigram.json"): continue
    os.makedirs(out_dir, exist_ok=True)
    
    word_count = pickle.load(open(f'python_inputs/{args.domain}_counts.pkl','rb'))
    sp_unigram = SentencePieceUnigramTokenizer(add_prefix_space=False)
    word_counter = word_count_iterator(word_count)
    sp_unigram.train_from_iterator(word_counter, vocab_size=vocab_size, show_progress=False, special_tokens=[])
    sp_unigram.save_model(out_dir)