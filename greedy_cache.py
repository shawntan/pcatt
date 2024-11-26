import argparse
import ctypes
import pickle
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, Array
from time import process_time
from tqdm import tqdm
import operator
from functools import reduce

"""
# Quick execute
nohup python -u greedy_cache.py --domain un -w 40 -k 5000 > logs/un2.log &
nohup python -u greedy_cache.py --domain arxiv -w 100 -k 5000 > logs/arxiv2.log &
nohup python -u greedy_cache.py --domain pubmed -w 100 -k 5000 > logs/pubmed2.log &
nohup python -u greedy_cache.py --domain wiki -w 100 -k 5000 > logs/wiki2.log &
"""


def get_inputs(counts):
    """Initial data setup for the greedy algorithm.

    Args:
        counts (Dict[str, int]): A mapping between words in $\bW$ and their counts

    Returns:
        List[str]: A list of words in $\bW$
        List[bytes]: A list of singletons ($\bSigma$)
        multiprossing.Array: Every word is concatenated together to form an array for easy access
        multiprossing.Array: Count score of word located at word_start_index
        multiprossing.Array: Tracks instances of tokens appearing multiple times in a word
        Dict[Tuple[bytes], List[str]]: A mapping between substrings in $\bS$ to words in $\bW$
        Dict[List[str], Tuple[bytes]]: A mapping between words in $\bW$ to substrings in $\bS$
        Dict[str, Pair[int]]:A mapping between words in $\bW$ to their start and end (not-inclusive) positions on the Array
        Dict[Pair[int], str]: A mapping between positions in Array to words in $\bW$
        Dict[Tuple[bytes], int]: Each substring in $\bS$ given an index
    """
    alphabet = set()
    arr_len = 0
    S2W = defaultdict(list)
    next_id = 0
    W2idx = {}
    idx2W = {}
    # find all possible substrings
    for w, word in tqdm(enumerate(counts), total=len(counts)):
        word = [bytes([b]) for b in word.encode("utf-8")]
        arr_len += len(word)
        for i in range(len(word)):
            end_id = next_id + len(word)
            alphabet.add(word[i])
            for j in range(i + 1, len(word) + 1):
                S2W[tuple(word[i:j])].append((w, i, j))

    print("bytes:", len(alphabet))
    print(len(S2W))
    for a in alphabet:
        if (a,) in S2W:
            del S2W[(a,)]
    print(len(S2W))

    shared_X_W = Array(ctypes.c_long, arr_len, lock=False)
    shared_C = Array(ctypes.c_long, arr_len, lock=False)
    shared_D = Array(ctypes.c_long, arr_len, lock=False)

    # let -1 represent singletons without cover
    for i in range(len(shared_X_W)):
        shared_X_W[i] = -1
    for w, word in enumerate(counts):
        b = [bytes([b]) for b in word.encode("utf-8")]
        end_id = next_id + len(b)
        W2idx[w] = (next_id, end_id)  # end_id not inclusive
        idx2W[(next_id, end_id)] = w
        shared_C[next_id] = counts[word]
        next_id = end_id

    S2W = {
        k: [(W2idx[w][0], W2idx[w][1], i, j) for w, i, j in v] for k, v in S2W.items()
    }
    W2S = {}
    for k, v in S2W.items():
        for w_start, w_end, i, j in v:
            w = (w_start, w_end)
            if w not in W2S:
                W2S[w] = set()
            W2S[w].add(k)
    print("substrings:", len(S2W))
    return (
        list(counts.keys()),
        alphabet,
        shared_X_W,
        shared_C,
        shared_D,
        S2W,
        W2S,
        W2idx,
        idx2W,
    )


def get_score_helper(inputs):
    """Calculate the score when selecting the substring in the current environment state

    Args:
        inputs (_type_): A pair consisting of a substring $S$, and its every position (start, end) on the array

    Returns:
        Pair[Tuple[bytes], int]: A pair consisting of a substring and the total number of merges it accomplished
    """
    substring, places = inputs
    counts = 0
    checked = defaultdict(list)
    for w_start, w_end, i, j in places:
        nones = 0
        uniqs = set()
        # if constraints are not valid, substring does not fully cover another chosen substring
        # constraint: starting singleton must not be covered by a token prior that spans across itself and its preceding singleton
        # First, we check if preceding singleton remains a singleton, we check for other conditions if its not a singleton
        # If preceding singleton has been covered by another token, we check if it spans and include starting singleton
        # Finally, multiple same tokens can occur sequentially, we check if the singletons are covered by one token.
        #   - e.g.  index[(i,n)] = 187 then ...,i,n,i,n,... represented as ...,187,187,187,187,...
        if (
            i > 0
            and X_W_arr[w_start + i - 1] != -1
            and X_W_arr[w_start + i - 1] == X_W_arr[w_start + i]
            and D_arr[w_start + i - 1] == D_arr[w_start + i]
        ):
            continue
        # repeat again for last singleton in substring
        if (
            w_start + j < w_end
            and X_W_arr[w_start + j] != -1
            and X_W_arr[w_start + j - 1] == X_W_arr[w_start + j]
            and D_arr[w_start + j - 1] == D_arr[w_start + j]
        ):
            continue
        # Check if we already considered merging parts of it
        # Example: a,b,a,b,a - (a,b,a) can only cover one instance
        if w_start in checked:
            add = True
            for i2, j2 in checked[w_start]:
                if (i >= i2 and i < j2) or (j >= i2 and j < j2):
                    add = False
                    continue
            if not add:
                continue
        # If substring cover is valid, add its contribution
        # Merges calculated from each unique token and each singleton covered by $S$
        for k in range(i, j):
            if X_W_arr[w_start + k] == -1:
                nones += 1
            else:
                uniqs.add((X_W_arr[w_start + k], D_arr[w_start + k]))
        counts += C_arr[w_start] * (nones + len(uniqs) - 1)
        checked[w_start].append((i, j))
    return substring, counts


def alter_graph(items, X_W_arr, D_arr, substring_idx):
    """After finding the best substring $S$, we want to update the current environment

    Args:
        items (List[int]): Each item is a quartet that represents the positions of the word and substring location
        X_W_arr (_type_): array to indicate positions of selected tokens, -1 for singletons
        D_arr (_type_): array to indicate positions of duplicates
        substring_idx (_type_): set position to token id of substring

    Returns:
        _type_: _description_
    """
    X_W = np.frombuffer(X_W_arr, dtype=np.int64)
    D_W = np.frombuffer(D_arr)
    visited = set()
    prev_w_start = None
    d_counter = 0
    for w_start, w_end, i, j in items:
        if (
            i > 0
            and X_W_arr[w_start + i - 1] != -1
            and X_W_arr[w_start + i - 1] == X_W_arr[w_start + i]
            and D_arr[w_start + i - 1] == D_arr[w_start + i]
        ):
            continue
        if (
            w_start + j < w_end
            and X_W_arr[w_start + j] != -1
            and X_W_arr[w_start + j - 1] == X_W_arr[w_start + j]
            and D_arr[w_start + j - 1] == D_arr[w_start + j]
        ):
            continue
        X_W[w_start + i : w_start + j] = substring_idx
        visited.add((w_start, w_end))

        if w_start != prev_w_start:
            d_counter = 0
        if w_start == prev_w_start:
            d_counter += 1

        D_W[w_start + i : w_start + j] = d_counter
        prev_w_start = w_start

    return visited


def init_globals(W_array, C_array, D_array):
    """
    Helper function to define global variables for using in Pool. (Python-Specific)
    Args:
        W_array (multiprossing.Array): Every word is concatenated together to form an array for easy access
        C_array (multiprossing.Array): Count score of word located at word_start_index
        D_array (multiprossing.Array): Tracks instances of tokens appearing multiple times in a word
    """
    global X_W_arr
    global C_arr
    global D_arr
    X_W_arr = W_array
    C_arr = C_array
    D_arr = D_array


def main():
    """
    # Quick execute
    nohup python -u greedy_cache.py --domain un -w 50 -k 5000 > logs/un2.log &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain")
    parser.add_argument(
        "-w", "--workers", dest="workers", type=int, metavar="int", default=50
    )
    parser.add_argument("-k", dest="k", type=int, metavar="int")
    args = parser.parse_args()

    count = pickle.load(open(f"python_inputs/{args.domain}_counts.pkl", "rb"))
    words, alpha, shared_X_W, shared_C, shared_D, S2W, W2S, W2idx, idx2W = get_inputs(
        count
    )

    ranks = {}  # selected tokens
    scores = []  # track contribution at each step
    shortlist = set(S2W.keys())  # substrings to be examined
    saved_merges = {}  # cache to save previous computations
    with Pool(
        args.workers,
        initializer=init_globals,
        initargs=(shared_X_W, shared_C, shared_D),
    ) as pool:
        for rank_id in range(args.k):
            start = process_time()

            # parallel compute (Python-specific)
            merges = pool.map(
                get_score_helper,  # [s for s in S2W.items() if s[0] in shortlist]
                [(s, S2W[s]) for s in shortlist],
            )
            # merges.extend(saved_merges)
            saved_merges.update({k: v for k, v in merges})

            # find best substring and its contribution
            merge_substring, score = max(
                saved_merges.items(), key=lambda x: (x[1], len(x[0]))
            )
            ranks[merge_substring] = rank_id
            scores.append(score)
            del saved_merges[merge_substring]

            # update current environment
            start2 = process_time()
            visited = alter_graph(
                S2W[merge_substring],
                shared_X_W,
                shared_D,
                substring_idx=rank_id,
            )

            # Update cache
            shortlist = set([token for v in visited for token in W2S[v]]) - set(
                ranks.keys()
            )
            print(
                reduce(operator.add, merge_substring, b""),
                f"({score}) | mp time: {start2-start:.4f}",
                f"misc time: {process_time()-start2:.4f} | saved: {len(saved_merges)}",
            )

    pickle.dump(
        {
            "tokens": ranks,
            "scores": scores,
            "partitions": np.ctypeslib.as_array(shared_X_W),
            "ranks": ranks,
        },
        open(f"python_outputs/{args.domain}_initial_cache.pkl", "wb"),
    )


if __name__ == "__main__":
    main()
