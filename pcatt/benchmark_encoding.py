import tiktoken

from pcatt import greedy_builder
import regex
import time

def process_wiki_xml(f):
    containers = []
    container = []
    for line in f:
        if line.startswith("<"):
            container = " ".join(container[1:])
            if len(container.split(" ")) >= 25:
                containers.append(container)
            container = []
            token_count = 0
            continue
        line = line.strip()
        if len(line) > 0:
            container.append(line)
    return containers

import tiktoken
cl100k_base = tiktoken.get_encoding("cl100k_base")
rules = list(cl100k_base._mergeable_ranks.keys())
orig = [ x for B in 'ABCDE' for i in range(10) for x in process_wiki_xml(open(f"/data/jiapeng/wiki/cleaned/A{B}/wiki_0{i}"))]
print("Number of texts:", len(orig),'\n')
pat_str = "'s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:punct:]]+| ?[^\\s[:alpha:][:punct:]]+|\\s+(?!\\S)|\\s+"
processed = [regex.findall(pat_str, doc) for doc in orig]
print('Words:', sum([len(x) for x in processed]))

times = 1

enc = tiktoken.Encoding(
    name="",
    pat_str=pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
print('tiktoken')
astart, tstart = time.time(), time.process_time()
for i in range(times):
    cl100k_base.encode_batch(orig, num_threads=8)
aend, tend = time.time()-astart, time.process_time() - tstart
print(f"Absolute time: {aend:.2f} second")
print(f"Absolute time: {sum([len(x) for x in processed])*times/aend:.2f} words/second")
print(f"Relative time: {tend:.2f} second")
print(f"Relative time: {sum([len(x) for x in processed])*times/tend:.2f} words/second")
print()


print('ours with regex')
gb = greedy_builder.build_greedy_tokenizer(rules)
gb.set_regex_pattern(pat_str)
astart, tstart = time.time(), time.process_time()
for i in range(times):
    tokenized = gb.batch_split_and_tokenize(orig)
aend, tend = time.time()-astart, time.process_time() - tstart
print(f"Absolute time: {aend:.2f} second")
print(f"Absolute time: {sum([len(x) for x in processed])*times/aend:.2f} words/second")
print(f"Relative time: {tend:.2f} second")
print(f"Relative time: {sum([len(x) for x in processed])*times/tend:.2f} words/second")
print()


print('ours without regex')
gb = greedy_builder.build_greedy_tokenizer(rules)
astart, tstart = time.time(), time.process_time()
for i in range(times):
    tokenized = gb.batch_tokenize_in_parts(processed)
aend, tend = time.time()-astart, time.process_time() - tstart
print(f"Absolute time: {aend:.2f} second")
print(f"Absolute time: {sum([len(x) for x in processed])*times/aend:.2f} words/second")
print(f"Relative time: {tend:.2f} second")
print(f"Relative time: {sum([len(x) for x in processed])*times/tend:.2f} words/second")
print()

print('ours new')
gb = greedy_builder.build_greedy_tokenizer(rules)
astart, tstart = time.time(), time.process_time()
for i in range(times):
    tokenized = gb.batch_tokenize_whole(orig)
aend, tend = time.time()-astart, time.process_time() - tstart
print(f"Absolute time: {aend:.2f} second")
print(f"Absolute time: {sum([len(x) for x in processed])*times/aend:.2f} words/second")
print(f"Relative time: {tend:.2f} second")
print(f"Relative time: {sum([len(x) for x in processed])*times/tend:.2f} words/second")
print()

