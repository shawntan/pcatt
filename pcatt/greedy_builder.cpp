#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <string>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include "tbb.h"
using namespace std;
namespace chrono = std::chrono;

/*
c++ -O3 -Wall -shared -std=c++20 \
-fPIC $(python3 -m pybind11 --includes) \
-I$CONDA_PREFIX/include/ \
-I$CONDA_PREFIX/include/tbb \
-I$CONDA_PREFIX/include/oneapi \
-L$CONDA_PREFIX/lib/ \
-l tbb \
./pcatt/greedy_builder.cpp \
-o ./pcatt/greedy_builder$(python3-config --extension-suffix) 
*/

struct SubstringPos
{
    long unsigned arr_start;
    long unsigned arr_end;
    unsigned int substr_start;
    unsigned int substr_end;
    SubstringPos() {};
    SubstringPos(long unsigned a, long unsigned b, unsigned int c, unsigned int d)
    {
        arr_start = a;
        arr_end = b;
        substr_start = c;
        substr_end = d;
    }
};

class Greedy_PCO_Tokenizer
{

public:
    static bool substring_pos_sorter(SubstringPos const &lhs, SubstringPos const &rhs)
    {
        return lhs.substr_start < rhs.substr_start;
    }
    
    long unsigned singleton_count = 0;
    const unordered_map<string, long unsigned> word_counts;
    const unordered_set<string> candidate_tokens;
    vector<string> ranks;
    vector<long unsigned> scores;
    unordered_set<string> shortlist;
    unordered_map<long unsigned, long unsigned> id_to_count;
    unordered_map<string, pair<long unsigned, long unsigned>>
        word_to_index;
    unordered_map<long unsigned, string> index_to_word;
    unordered_map<string, vector<SubstringPos>> substring_to_index;
    unordered_map<string, unordered_set<string>> word_to_substring;
    vector<int unsigned> T_arr;
    vector<int unsigned> D_arr;
    unordered_map<string, long unsigned> results;

    Greedy_PCO_Tokenizer(unordered_map<string, long unsigned> &word_counts, unordered_set<string> &candidate_tokens)
    : word_counts(word_counts), candidate_tokens(candidate_tokens)
    {}

    virtual ~Greedy_PCO_Tokenizer() {}

    void initialize_graph() {
        cout << "Word counts size: " << word_counts.size() << endl;
        cout << "Token set size: " << candidate_tokens.size() << endl;
        if (candidate_tokens.size() == 0)
        {
            cout << "Empty token set size selected -> all possible substrings..." <<endl;
        }
        /* Initialize variables */
        auto start = chrono::high_resolution_clock::now();
        long unsigned next_id = 0;
        long unsigned end_id = 0;

        /* Initialize array positions */
        for (auto &item : word_counts)
        {

            singleton_count += item.first.size();

            end_id = next_id + item.first.size();
            id_to_count[next_id] = item.second;

            word_to_index[item.first] = pair(next_id, end_id);
            word_to_substring[item.first] = unordered_set<string>();
            for (unsigned int i = 0; i < item.first.size(); ++i)
            {

                for (unsigned int j = i + 1; j < item.first.size() + 1; ++j)
                {
                    string substr = item.first.substr(i, j - i);
                    if (substr.size() <= 1)
                    {
                        continue;
                    }
                    if (candidate_tokens.size() > 0 && candidate_tokens.find(substr) == candidate_tokens.end())
                    {
                        continue;
                    }
                    if (substring_to_index.find(substr) == substring_to_index.end())
                    {
                        substring_to_index[substr] = vector<SubstringPos>();
                    }
                    substring_to_index[substr].push_back({next_id, end_id, i, j});
                    word_to_substring[item.first].insert(substr);
                }
            }
            next_id = end_id;
        }

        for (auto &kv : word_to_index)
        {
            index_to_word[kv.second.first] = kv.first;
        }

        if (candidate_tokens.size() == 0)
        {
            cout << "Final token set size: " << substring_to_index.size() << endl;
        }

        /* initialize more variables */
        T_arr = vector<int unsigned>(singleton_count, 0);
        D_arr = vector<int unsigned>(singleton_count, 0);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Initial setup phase: " << duration.count() << " ms" << endl;
    }

    unsigned long get_singleton_counts() {
        return singleton_count;
    }

    unsigned long get_candidate_token_size(){
        if (candidate_tokens.size() == 0) {
            return substring_to_index.size();
        } else {
            return candidate_tokens.size();
        }
    }

    long unsigned get_score_helper(const vector<SubstringPos> &places, const vector<int unsigned> *T_arr_ptr, const vector<int unsigned> *D_arr_ptr, const unordered_map<long unsigned, long unsigned> &id_to_count)
    {
        long unsigned counts = 0;

        unordered_map<long unsigned, vector<SubstringPos>> pplaces;
        for (auto p : places)
        {
            if (pplaces.find(p.arr_start) == pplaces.end())
            {
                pplaces[p.arr_start] = vector<SubstringPos>();
            }
            pplaces[p.arr_start].push_back(p);
        }

        for (auto pp : pplaces)
        {
            long unsigned prev_end = 0;
            sort( // execution::par,
                pp.second.begin(), pp.second.end(), &substring_pos_sorter);

            for (auto &p : pp.second)
            {

                const long unsigned ws = p.arr_start;
                const long unsigned we = p.arr_end;
                const int i = p.substr_start;
                const int j = p.substr_end;
                if (ws + i < prev_end)
                {
                    continue;
                }
                if (i > 0 && (*T_arr_ptr)[ws + i - 1] != 0 && (*T_arr_ptr)[ws + i - 1] == (*T_arr_ptr)[ws + i] && (*D_arr_ptr)[ws + i - 1] == (*D_arr_ptr)[ws + i])
                {
                    continue;
                }
                if (ws + j < we && (*T_arr_ptr)[ws + j] != 0 && (*T_arr_ptr)[ws + j - 1] == (*T_arr_ptr)[ws + j] && (*D_arr_ptr)[ws + j - 1] == (*D_arr_ptr)[ws + j])
                {
                    continue;
                }
                int nones = 0;
                set<pair<int unsigned, int unsigned>> uniqs;
                for (int k = i; k < j; ++k)
                {
                    if ((*T_arr_ptr)[ws + k] == 0)
                    {
                        nones += 1;
                    }
                    else
                    {
                        uniqs.insert(pair((*T_arr_ptr)[ws + k], (*D_arr_ptr)[ws + k]));
                    }
                }
                counts += id_to_count.at(ws) * (nones + uniqs.size() - 1);
                prev_end = ws + j;
            }
        }
        return counts;
    }

    unordered_set<long unsigned> alter_graph(const vector<SubstringPos> &items, vector<int unsigned> *T_arr_ptr, vector<int unsigned> *D_arr_ptr, const int &substring_idx)
    {

        unordered_set<long unsigned> visited;
        long unsigned prev_w_start = -1;
        int d_counter = 0;
        for (auto &p : items)
        {
            const long unsigned ws = p.arr_start;
            const long unsigned we = p.arr_end;
            const int i = p.substr_start;
            const int j = p.substr_end;

            if (i > 0 && (*T_arr_ptr)[ws + i - 1] != 0 && (*T_arr_ptr)[ws + i - 1] == (*T_arr_ptr)[ws + i] && (*D_arr_ptr)[ws + i - 1] == (*D_arr_ptr)[ws + i])
            {
                continue;
            }
            if (ws + j < we && (*T_arr_ptr)[ws + j] != 0 && (*T_arr_ptr)[ws + j - 1] == (*T_arr_ptr)[ws + j] && (*D_arr_ptr)[ws + j - 1] == (*D_arr_ptr)[ws + j])
            {
                continue;
            }

            visited.insert(ws);
            if (ws != prev_w_start)
            {
                d_counter = 0;
            }
            if (ws == prev_w_start)
            {
                d_counter += 1;
            }
            for (long unsigned k = ws + i; k < ws + j; ++k)
            {
                (*T_arr_ptr)[k] = substring_idx;
                (*D_arr_ptr)[k] = d_counter;
            }
            prev_w_start = ws;
        }
        return visited;
    }

    pair<vector<string>, vector<long unsigned>> custom_steps(vector<string> &starting_tokens){
        for (string &token : starting_tokens)
        {
            unsigned int rank = ranks.size();
            ranks.push_back(token);
            unsigned long score = get_score_helper(substring_to_index[token], &T_arr, &D_arr, id_to_count);
            scores.push_back(score);
            alter_graph(substring_to_index[token], &T_arr, &D_arr, rank);
            cout << rank << ". |" << token << " [" << hex;
            for (auto c : token)
            {
                cout << (unsigned int)(unsigned char)c << " ";
            }
            cout << dec << "] | " << score << endl;
        }
            for (auto &r : ranks)
            {
                shortlist.erase(r);
                results.erase(r);
            }
        return pair(ranks, scores);
    }

    pair<vector<string>, vector<long unsigned>> solve_to_step(unsigned int k)
    {
        auto total_start = chrono::high_resolution_clock::now();

        for (auto &s : substring_to_index)
        {
            shortlist.insert(s.first);
        }
        for (auto &s : shortlist)
        {
            results[s] = 0;
        }

        /* Main GreedTok routine */
        cout << "Starting main routine..." << endl;
        for (unsigned int rank = ranks.size() + 1; rank <= k; ++rank)
        {
            auto start = chrono::high_resolution_clock::now();

            vector<string> items(shortlist.begin(), shortlist.end());
            oneapi::tbb::parallel_for(tbb::blocked_range<long unsigned>(0, items.size()), [&](tbb::blocked_range<long unsigned> r)
                                      { for (long unsigned i=r.begin(); i<r.end(); ++i){
                    results[items[i]] = get_score_helper(substring_to_index[items[i]], &T_arr, &D_arr, id_to_count); } });

            pair<string, long unsigned> best = *max_element(results.begin(), results.end(), [](const pair<string, long unsigned> a, const pair<string, long unsigned> b)
                                                            { return a.second < b.second; });
            ranks.push_back(best.first);
            scores.push_back(best.second);

            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
            unordered_set<long unsigned> visited = alter_graph(substring_to_index[best.first], &T_arr, &D_arr, rank);

            shortlist.clear();
            for (auto &v : visited)
            {
                shortlist.insert(word_to_substring[index_to_word[v]].begin(),
                                 word_to_substring[index_to_word[v]].end());
            }
            for (auto &r : ranks)
            {
                shortlist.erase(r);
            }
            results.erase(best.first);

            stop = chrono::high_resolution_clock::now();
            auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop - start);
            cout << rank << ". |" << best.first << " [" << hex;
            for (auto c : best.first)
            {
                cout << (unsigned int)(unsigned char)c << " ";
            }
            cout << dec << "] | " << best.second << " | " << duration.count() << " ms | " << duration2.count() << " ms | shortlist: " << shortlist.size() << endl;
        }
        auto total_duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - total_start);
        cout << "Total time taken: " << total_duration.count() << " seconds" << endl;
        return pair(ranks, scores);
    }

};

class Greedy_Tokenizer
{
    const vector<string> rules;
    unordered_map<string, int unsigned> rules_cache;

    vector<int unsigned> flatten(vector<vector<int unsigned>> &v)
    {
        vector<int unsigned> r;
        for (auto &i : v)
        {
            r.insert(r.end(), i.begin(), i.end());
        }
        return r;
    }

public:
    Greedy_Tokenizer(vector<string> &rules)
        : rules(rules)
    {
        for (int unsigned i = 0; i < rules.size(); i++)
        {
            rules_cache[rules.at(i)] = 256 + i;
        }
    }
    virtual ~Greedy_Tokenizer() {}

    vector<int unsigned> tokenize(const string &word)
    {
        if (rules_cache.find(word) != rules_cache.end())
        {
            return vector<int unsigned>{rules_cache.at(word)};
        }
        vector<int unsigned> result;
        vector<int unsigned> T_arr(word.size(), 0);
        vector<int unsigned> D_arr(word.size(), 0);

        for (int unsigned r = 1; r <= rules.size(); r++)
        {
            string substr = rules.at(r - 1);
            int unsigned substr_size = substr.size();
            int unsigned d_counter = 0;

            if (substr_size > word.size())
            {
                continue;
            }

            for (int unsigned i = 0; i < word.size() - substr_size + 1; i++)
            {
                int unsigned j = i + substr_size;
                if (substr[0] != word[i] || substr[substr_size - 1] != word[j - 1])
                {
                    continue;
                }
                if (substr_size > 2 && !equal(word.begin() + i, word.begin() + j, substr.begin()))
                {
                    continue;
                }
                if (i > 0 && T_arr[i - 1] != 0 && T_arr[i - 1] == T_arr[i] && D_arr[i - 1] == D_arr[i])
                {
                    continue;
                }
                if (j < word.size() && T_arr[j] != 0 && T_arr[j - 1] == T_arr[j] && D_arr[j - 1] == D_arr[j])
                {
                    continue;
                }

                d_counter += 1;

                for (int unsigned k = i; k < j; ++k)
                {
                    T_arr[k] = r;
                    D_arr[k] = d_counter;
                }
                i += substr_size - 1;
            }
        }

        for (int unsigned i = 0; i < T_arr.size(); i++)
        {
            if (T_arr[i] == 0)
            {
                result.push_back((int unsigned)(char unsigned)word[i]);
            }
            else
            {
                result.push_back(255 + T_arr[i]);
                i += rules.at(T_arr[i] - 1).size() - 1;
            }
        }
        return result;
    }

    vector<int unsigned> tokenize(vector<string> &sentence)
    {
        vector<vector<int unsigned>> results(sentence.size(), vector<int unsigned>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, sentence.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize(sentence.at(i)); } });
        return flatten(results);
    }

    vector<vector<int unsigned>> batch_tokenize(vector<vector<string>> &sentences)
    {
        vector<vector<int unsigned>> results(sentences.size(), vector<int unsigned>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, sentences.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize(sentences.at(i)); } });
        return results;
    }

    vector<pair<int unsigned, int unsigned>> score_merges_per_turn(const string &word)
    {
        vector<pair<int unsigned, int unsigned>> result;
        vector<int unsigned> T_arr(word.size(), 0);
        vector<int unsigned> D_arr(word.size(), 0);

        for (int unsigned r = 1; r <= rules.size(); r++)
        {
            string substr = rules.at(r - 1);
            int unsigned substr_size = substr.size();
            int unsigned d_counter = 0;

            if (substr_size > word.size())
            {
                continue;
            }

            for (int unsigned i = 0; i < word.size() - substr_size + 1; i++)
            {
                int unsigned j = i + substr_size;
                if (substr[0] != word[i] || substr[substr_size - 1] != word[j - 1])
                {
                    continue;
                }
                if (substr_size > 2 && !equal(word.begin() + i, word.begin() + j, substr.begin()))
                {
                    continue;
                }
                if (i > 0 && T_arr[i - 1] != 0 && T_arr[i - 1] == T_arr[i] && D_arr[i - 1] == D_arr[i])
                {
                    continue;
                }
                if (j < word.size() && T_arr[j] != 0 && T_arr[j - 1] == T_arr[j] && D_arr[j - 1] == D_arr[j])
                {
                    continue;
                }

                d_counter += 1;
                set<pair<int unsigned, int unsigned>> uniqs;
                int nones = 0;
                for (int unsigned k = i; k < j; ++k)
                {
                    if (T_arr[k] != 0)
                    {
                        uniqs.insert(pair(T_arr[k], D_arr[k]));
                    }
                    else
                    {
                        nones += 1;
                    }
                    T_arr[k] = r;
                    D_arr[k] = d_counter;
                }
                result.push_back(pair(r - 1, nones + uniqs.size() - 1));
                i += substr_size - 1;
            }
        }
        return result;
    }

    vector<vector<pair<int unsigned, int unsigned>>> batch_score_merges_per_turn(vector<string> &words)
    {
        vector<vector<pair<int unsigned, int unsigned>>> results(words.size(), vector<pair<int unsigned, int unsigned>>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = score_merges_per_turn(words[i]); } });
        return results;
    }

    int unsigned score_max_merges(const string &word)
    {
        return tokenize(word).size();
    }

    vector<int unsigned> batch_score_max_merges(vector<string> &words)
    {
        vector<int unsigned> results(words.size(), 0);
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize(words[i]).size(); } });
        return results;
    }

    int unsigned score_max_cover(const string &word)
    {
        vector<bool> T_arr(word.size() - 1, 0);
        int unsigned it, substr_size;
        for (int unsigned i = 0; i < rules.size(); i++)
        {
            string substr = rules.at(i);
            it = word.find(substr);
            substr_size = substr.size();

            while (it != string::npos)
            {
                for (int unsigned j = it; j < it; j++)
                {
                    T_arr[j] = true;
                }
                it = word.substr(it + substr_size).find(substr);
            }
        }
        return accumulate(T_arr.begin(), T_arr.end(), 0);
    }

    vector<int unsigned> batch_score_max_cover(vector<string> &words)
    {
        vector<int unsigned> results(words.size(), 0);
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = score_max_cover(words[i]); } });
        return results;
    }
};

class PyGreedy_PCO_Tokenizer : public Greedy_PCO_Tokenizer
{
public:
    using Greedy_PCO_Tokenizer::custom_steps;
    using Greedy_PCO_Tokenizer::solve_to_step;
    using Greedy_PCO_Tokenizer::get_score_helper;
    using Greedy_PCO_Tokenizer::initialize_graph;
    using Greedy_PCO_Tokenizer::get_singleton_counts;
    using Greedy_PCO_Tokenizer::get_candidate_token_size;
};

Greedy_PCO_Tokenizer *build_greedy_pco_tokenizer(unordered_map<string, long unsigned> &word_counts, unordered_set<string> candidate_tokens = {})
{
    return new Greedy_PCO_Tokenizer(word_counts, candidate_tokens);
}

class PyGreedy_Tokenizer : public Greedy_Tokenizer
{
public:
    using Greedy_Tokenizer::batch_score_max_cover;
    using Greedy_Tokenizer::batch_score_max_merges;
    using Greedy_Tokenizer::batch_score_merges_per_turn;
    using Greedy_Tokenizer::batch_tokenize;
    using Greedy_Tokenizer::Greedy_Tokenizer;
    using Greedy_Tokenizer::score_max_cover;
    using Greedy_Tokenizer::score_max_merges;
    using Greedy_Tokenizer::score_merges_per_turn;
    using Greedy_Tokenizer::tokenize;
};


Greedy_Tokenizer *build_greedy_tokenizer(vector<string> rules)
{
    return new Greedy_Tokenizer(rules);
}

PYBIND11_MODULE(greedy_builder, var)
{
    var.doc() = "greedy module";
    py::class_<Greedy_PCO_Tokenizer, PyGreedy_PCO_Tokenizer>(var, "Greedy_PCO_Tokenizer")
        .def(py::init<>([](unordered_map<string, long unsigned> &word_counts, 
        unordered_set<string> candidate_tokens = {})
            { return new Greedy_PCO_Tokenizer(word_counts, candidate_tokens); }))
        .def("solve_to_step", &Greedy_PCO_Tokenizer::solve_to_step)
        .def("get_score_helper", &Greedy_PCO_Tokenizer::get_score_helper)
        .def("initialize_graph", &Greedy_PCO_Tokenizer::initialize_graph)
        .def("alter_graph", &Greedy_PCO_Tokenizer::alter_graph)
        .def("custom_steps", &Greedy_PCO_Tokenizer::custom_steps)
        .def("get_singleton_counts", &Greedy_PCO_Tokenizer::get_singleton_counts)
        .def("get_candidate_token_size", &Greedy_PCO_Tokenizer::get_candidate_token_size);

    var.def("build_greedy_pco_tokenizer", &build_greedy_pco_tokenizer, "Factory function for greedy PCO tokenizer");

    py::class_<Greedy_Tokenizer, PyGreedy_Tokenizer>(var, "Greedy_Tokenizer")
        .def(py::init<>([](vector<string> &cover_order)
                        { return new Greedy_Tokenizer(cover_order); }))
        .def("tokenize", py::overload_cast<vector<string> &>(&Greedy_Tokenizer::tokenize))
        .def("tokenize", py::overload_cast<const string &>(&Greedy_Tokenizer::tokenize))
        .def("batch_tokenize", &Greedy_Tokenizer::batch_tokenize)
        .def("score_max_merges", &Greedy_Tokenizer::score_max_merges)
        .def("batch_score_max_merges", &Greedy_Tokenizer::batch_score_max_merges)
        .def("score_max_cover", &Greedy_Tokenizer::score_max_cover)
        .def("batch_score_max_cover", &Greedy_Tokenizer::batch_score_max_cover)
        .def("score_merges_per_turn", &Greedy_Tokenizer::score_merges_per_turn)
        .def("batch_score_merges_per_turn", &Greedy_Tokenizer::batch_score_merges_per_turn);
    var.def("build_greedy_tokenizer", &build_greedy_tokenizer, "Factory function for greedy tokenizer");
}
