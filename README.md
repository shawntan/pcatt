# An optimization approach to tokenization

### Greedy Approximate Solution
1. Install dependencies for C++ code, we use oneTBB to parallelize the code:
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh
sudo sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --cli
cd <install_dir>
```
2. Initialize environment variables:
```
cd <install_dir>
. ./oneapi/tbb/latest/env/vars.sh
```
3. Compile greedy_cache.cpp:
```
c++ -std=c++20 -o greedy.exe greedy_cache.cpp -ltbb -O3
```
4. Prepare inputs (refer to [cpp_inputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_inputs) for examples):
    * counts: a file with '\n' delimited integers
    * words: a file with ' ' (space) delimited words
5. Run compiled program
    * currently looks for domain inputs in fixed path under cpp_inputs/*
    * To-do: pybind11 implementation
```
./greedy.exe <domain> <k>
```

6. Now we obtained our ranked token sequence (refer to [cpp_outputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_outputs/) for examples):
    * merges: the number of merges at each step, delimited by '\n'
    * tokens: byte sequences in hex-format, delimited by '\n'
        * If only valid byte sequences are required, we have to prune the candidate token space [To-do]
        * Current implementation sees every possible substring

### Using the obtained ranked token sequence
To use the tokenizer, we also need the previous oneTBB dependency.
1. Additionally, install pybind11 dependency, simply:
```
pip3 install pybind11
```
2. Compile greedy_builder.cpp
```
c++ -O3 -Wall -shared -std=c++20 -ltbb -fPIC $(python3 -m pybind11 --includes) greedy_builder.cpp -o greedy_builder$(python3-config --extension-suffix)
```
3. Import in python
   
Examples in [eval_tokenizer_example.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_tokenizer_example.ipynb)

Evaluations in [eval_notebook.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_notebook.ipynb)

### Citation
TBD
