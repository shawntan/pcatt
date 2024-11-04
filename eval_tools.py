import pickle
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def get_merge_count(item, tokens):
    """Helper function to parallelize counting of merges

    Args:
        item (Pair[str, int]): word and its count
        tokens (List[Tuple[bytes]]): The order of tokens to merge

    Returns:
        str: word
        List[int]: A list of merges added at step $k$
        List[int]: A list of cumulative sum of merges at step $k$
        array[int]: token used at position in array
        array[int]: duplicate tokens used at position in array
    """
    rank = {token: i for i, token in enumerate(tokens)}
    w, count = item
    b = [bytes([byte]) for byte in w.encode("utf-8")]
    arr = np.zeros(len(b)) - 1
    dup = np.zeros(len(b))

    total_scores = []
    add_merges = []
    partitions = []

    total_counter = 0
    for _id, token in enumerate(tokens):
        j = len(token)
        counter = 0
        i = 0

        if len(token) == 1 or len(arr) < j:
            add_merges.append(counter)
            total_scores.append(total_counter)
            partitions.append(partitions[-1] if len(partitions) > 0 else len(w) * count)
            continue

        if len(arr) == j and tuple(b) == token:
            nones = 0
            uniqs = set()
            for k in range(len(arr)):
                if arr[k] == -1:
                    nones += 1
                else:
                    uniqs.add(arr[k])
            arr[:] = rank[token]
            counter += count * (nones + len(uniqs) - 1)
        else:
            dup_counter = 0
            while i <= len(arr) - j:
                if b[i] != token[0]:
                    i += 1
                    continue
                elif b[i + j - 1] != token[-1]:
                    i += 1
                    continue
                elif tuple(b[i : i + j]) != token:
                    i += 1
                    continue
                # word partitions match token, proceed check array
                if i > 0:
                    if (
                        arr[i - 1] != -1
                        and arr[i - 1] == arr[i]
                        and dup[i - 1] == dup[i]
                    ):
                        i += 1
                        continue
                if i + j < len(arr):
                    if (
                        arr[i + j] != -1
                        and arr[i + j - 1] == arr[i + j]
                        and dup[i + j - 1] == dup[i + j]
                    ):
                        i += 1
                        continue

                nones = 0
                uniqs = set()
                for k in range(i, i + j):
                    if arr[k] == -1:
                        nones += 1
                    else:
                        uniqs.add(arr[k])
                arr[i : i + j] = rank[token]
                dup[i : i + j] = dup_counter
                counter += count * (nones + len(uniqs) - 1)
                dup_counter += 1
                i += len(token)

        parts = 1
        for i in range(len(arr) - 1):
            if arr[i] == -1 or arr[i] != arr[i + 1] or dup[i] != dup[i + 1]:
                parts += 1

        total_counter += counter
        total_scores.append(total_counter)
        add_merges.append(counter)
        partitions.append(parts * count)

    return w, add_merges, total_scores, partitions#, arr, dup


def get_merge_count_main(tokens, count):
    """Main routine to calculate total merges.

    Args:
        tokens (List[Tuple[bytes]]): The order of tokens to merge
        count (Dict[word, int]): Map of word count scores

    Returns:
        List[int]: A list of merges added at step $k$
        List[int]: A list of cumulative sum of merges at step $k$
        Dict[str, array[int]]: Map of word to token used at position in array
        Dict[str, array[int]]: Map of word to duplicate tokens used at position in array
    """
    sum_qty_words = sum(count.values())
    with Pool(30) as p:
        outputs = p.map(partial(get_merge_count, tokens=tokens), tqdm(count.items()))
    add_merges = np.array([o[1] for o in outputs]).sum(axis=0)
    total_scores = np.array([o[2] for o in outputs]).sum(axis=0)
    total_partitions = np.array([o[3] for o in outputs]).sum(axis=0)/sum_qty_words
    # word2part = {o[0]: o[-2] for o in outputs}
    # word2dup = {o[0]: o[-1] for o in outputs}
    return add_merges, total_scores, total_partitions #word2part, word2dup

def calc_token_per_word(items):
    """Helper function to calculate token per word

    Args:
        items (_type_): A triplet tuple of:
            str: word
            array: denoting positions of token coverage
            array: denoting positions of duplicate tokens

    Returns:
        Pair[str, int]: word, and number of partitions of words
    """
    w, part, dup = items
    partitions = 1
    for i in range(len(part)-1):
        if part[i] == -1 or part[i] != part[i+1] or dup[i] != dup[i+1]:
            partitions+=1
    return w, partitions

def calc_token_per_word_main(word2part, word2dup, counts):
    """Main routine to calculate token per word.
    Intended for direct evaluation of MIP solution (if implemented).

    Args:
        word2part (Dict[str,array]): word-array mapping denoting token covering position
        word2dup (Dict[str,array])): word-array mapping denoting duplicate tokens
        counts (Dict[str,int]): word-count mapping

    Returns:
        _type_: _description_
    """
    with Pool(30) as p:
        outputs = p.map(calc_token_per_word,
                        tqdm([(w, k, word2dup[w]) for w,k in word2part.items()]))
    outputs = {w:k for w,k in outputs}
    total_parts = sum([outputs[w]*counts[w] for w in counts])
    total_words = sum(list(counts.values()))
    return total_parts/total_words

def example():
    """Example code for running evaluations

    Args:
        domain (_type_): _description_
    """

    domain = "un"
    counts = pickle.load(open(f"python_inputs/{domain}_counts.pkl", "rb"))

    # greedy
    tokens = pickle.load(open(f"outputs/{domain}_initial_cache.pkl", "rb"))["tokens"]

    # BPE
    # tokens = []
    # bpe_db = pickle.load(open(f"python_outputs/{domain}_bpe.pkl",'rb'))
    # for p1, p2 in bpe_db['rules']:
    #     token = []
    #     token.extend([bytes([b]) for b in p1])
    #     token.extend([bytes([b]) for b in p2])
    #     tokens[domain].append(tuple(token))

    start, stop = 0, 5000
    (
        merges,
        scores,
        tokens_per_word,
        word2part,
        word2dup,
    ) = get_merge_count_main(tokens[start:stop], counts)
