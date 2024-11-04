import pickle
import collections
import argparse

"""
The code here is modified from the snippet on HuggingFace.
The runtime performance is bad (for python) from lack of parallelization but it works.

# Quick execute
nohup python bpe.py --domain arxiv > logs/arxiv.log &
nohup python bpe.py --domain pubmed > logs/pubmed.log &
nohup python bpe.py --domain un > logs/un.log &
nohup python bpe.py --domain wiki > logs/wiki.log &

Running the code for these scenarios is not recommended as it take days to complete.
"""


def bpe_train(
    data: dict[str, int], vocab_size: int,) -> dict[bytes, int]:
    
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    rules = []
    for i in range(2**8):
        ranks[bytes([i])] = i

    words: dict[str, list[bytes]] = {
        word: [bytes([b]) for b in word.encode("utf-8")] 
        for word in data
    }
    print(len(words))

    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Find the most common pair. This will become our next token
        stats = collections.Counter()
        for word, pieces in words.items():
            for pair in zip(pieces[:-1], pieces[1:]):
                stats[pair] += data[word]

        most_common_pair = max(stats, key=lambda x: stats[x])
        rules.append(most_common_pair)
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        ranks[token_bytes] = token

        new_words = {}
        for word, pieces in words.items():
            new_pieces = []
            i = 0
            while i < len(pieces) - 1:
                if (pieces[i], pieces[i + 1]) == most_common_pair:
                    new_pieces.append(token_bytes)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            if i == len(pieces) - 1:
                new_pieces.append(pieces[i])
            new_words[word] = new_pieces
        words = new_words
        print(len(ranks))
    return {'tokens':ranks, 'rules':rules}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain")
    args = parser.parse_args()

    word_counts = pickle.load(open(f'./python_inputs/{args.domain}_counts.pkl', 'rb'))
    output = bpe_train(word_counts, vocab_size=32000)
    pickle.dump(output, open(f'./python_outputs/slow_bpe/{args.domain}_bpe.pkl','wb'))
    
if __name__ == 'main':
    main()
