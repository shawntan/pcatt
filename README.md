# A partition cover approach to tokenization

### GreedTok 
1. Install dependencies for C++ code, we use oneTBB to parallelize the code, simplest way is to use Conda:
```
conda install tbb-devel
```
2. If using python wrapper (Todo: automate pip installation)
   
    a. Using pip:
      ```
      pip install -r requirements.txt
      pip install .
      ```
    b. Or compile manually e.g. (have to specify links)
      ```
      c++ -O3 -Wall -shared -std=c++20 \
      -fPIC $(python3 -m pybind11 --includes) \
      -I$CONDA_PREFIX/include/ \
      -I$CONDA_PREFIX/include/tbb \
      -I$CONDA_PREFIX/include/oneapi \
      -L$CONDA_PREFIX/lib/ \
      -l tbb \
      ./pcatt/greedy_builder.cpp \
      -o ./pcatt/greedy_builder$(python3-config --extension-suffix) 
      ```
    c. import and use! Examples in [eval_tokenizer_example.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_tokenizer_example.ipynb)
3. If using C++ files directly

    a. Compile greedy_cache.py e.g.:
      ```
      c++ -O3 -std=c++20 \
      -I$CONDA_PREFIX/include/ \
      -I$CONDA_PREFIX/include/tbb \
      -I$CONDA_PREFIX/include/oneapi \
      -L$CONDA_PREFIX/lib/ \
      -l tbb \
      pcatt/greedy_cache.cpp \
      -o pcatt/greedy.exe 
      ```
    b. Prepare inputs (refer to [cpp_inputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_inputs) for examples):
      * counts: a file with '\n' delimited integers
      * words: a file with ' ' (space) delimited words
        
    c. Run compiled program (currently looks for domain inputs in fixed path under cpp_inputs/*)
        ```
         ./greedy.exe <domain> <k>
        ```
    d. Now we obtained our ranked token sequence (refer to [cpp_outputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_outputs/) for examples):
      * merges: the number of covers at each step, delimited by '\n'
      * tokens: byte sequences in hex-format, delimited by '\n'

Evaluations in [eval_notebook.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_notebook.ipynb)

### Citation
TBD
