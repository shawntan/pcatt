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
#include <regex>
#include <limits.h>
#include "tbb.h"
using namespace std;
namespace chrono = std::chrono;

/*
c++ -O3 -Wall -shared -std=c++20 \
-fPIC $(python3 -m pybind11 --includes) \
-I$CONDA_PREFIX/include/ \
-I$CONDA_PREFIX/include/tbb \
-L$CONDA_PREFIX/lib/ \
-l tbb \
./pcatt/pco_tokenizer.cpp \
-o ./pcatt/pco_tokenizer$(python3-config --extension-suffix)
*/

struct SubstringPos
{
    long unsigned arr_start;
    long unsigned arr_end;
    unsigned int substr_start;
    unsigned int substr_end;
    /**
     * @brief Construct a new Substring Pos object, meant for internal use
     *
     * @param a index of start of word in array
     * @param b index of end of word in array
     * @param c start of substring in word
     * @param d end of substring in word
     */
    SubstringPos(long unsigned a, long unsigned b, unsigned int c, unsigned int d)
    {
        arr_start = a;
        arr_end = b;
        substr_start = c;
        substr_end = d;
    }
};

class GreedyPCOTokenizer
{

public:
    /**
     * @brief custom function to sort SubstringPos, meant for internal use
     *
     * @param lhs SubstringPos 1
     * @param rhs SubstringPos 2
     * @return true when lhs < rhs based on start index
     * @return false otherwise
     */
    static bool substring_pos_sorter(
        SubstringPos const &lhs,
        SubstringPos const &rhs)
    {
        return lhs.substr_start < rhs.substr_start;
    }

    long unsigned singleton_count = 0;
    unordered_map<string, long unsigned> word_counts;
    unordered_set<string> candidate_tokens;
    vector<string> ranks;
    vector<long unsigned> scores;
    unordered_set<string> shortlist;
    unordered_map<long unsigned, long unsigned> id_to_count;
    unordered_map<string, pair<long unsigned, long unsigned>> word_to_index;
    unordered_map<long unsigned, string> index_to_word;
    unordered_map<string, vector<SubstringPos>> substring_to_index;
    unordered_map<string, unordered_set<string>> word_to_substring;
    vector<int unsigned> T_arr;
    vector<int unsigned> D_arr;
    unordered_map<string, long unsigned> results;

    /**
     * @brief Construct a new Greedy P C O Tokenizer object
     *
     * @param word_counts word to count mapping
     * @param candidate_tokens to investigate
     */
    GreedyPCOTokenizer(
        unordered_map<string, long unsigned> word_counts = {},
        unordered_set<string> candidate_tokens = {})
        : word_counts(word_counts), candidate_tokens(candidate_tokens)
    {
    }

    virtual ~GreedyPCOTokenizer() {}

    void build_counter_from_text(const vector<vector<string>> &texts)
    {

        tbb::concurrent_hash_map<string, unsigned long> async_counter;
        tbb::parallel_for(
            tbb::blocked_range<long unsigned>(0, texts.size()),
            [&](tbb::blocked_range<long unsigned> r)
            {
                unordered_map<string, unsigned long> temp_counter;
                for (long unsigned i = r.begin(); i < r.end(); ++i)
                {
                    for (string w : texts.at(i))
                    {
                        auto p = temp_counter.try_emplace(w, 0);
                        p.first->second += 1;
                    }
                }
                tbb::concurrent_hash_map<string, unsigned long>::accessor a;
                for (auto &item : temp_counter)
                {
                    async_counter.insert(a, item.first);
                    a->second += item.second;
                    a.release();
                }
            });
        for (auto item : async_counter)
        {
            word_counts.emplace(item.first, item.second);
        }
    }

    /**
     * @brief Create a bipartite graph representation and allocate spaces for tracking arrays
     */
    void initialize_graph(
        size_t max_token_length = UINT8_MAX,
        unsigned int min_word_count = 1)
    {
        cout << "Word counts size: " << word_counts.size() << endl;
        cout << "Token set size: " << candidate_tokens.size() << endl;
        if (candidate_tokens.size() == 0)
        {
            cout << "Empty token set size selected -> all possible substrings with..." << endl;
        }
        cout << "Max token size: " << max_token_length << endl;
        cout << "Min. word count: " << min_word_count << endl;
        /* Initialize variables */
        auto start = chrono::high_resolution_clock::now();
        long unsigned next_id = 0;
        long unsigned end_id = 0;

        /* Initialize array positions */
        for (auto &item : word_counts)
        {
            if (item.second < min_word_count)
            {
                continue;
            }

            singleton_count += item.first.size();

            end_id = next_id + item.first.size();
            id_to_count[next_id] = item.second;

            word_to_index[item.first] = pair(next_id, end_id);
            word_to_substring[item.first] = unordered_set<string>();

            for (unsigned int i = 0; i < item.first.size(); ++i)
            {

                for (unsigned int j = i + 2; j < min(max_token_length + i, item.first.size() + 1); ++j)
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
            cout << "Final candidate token set size: " << substring_to_index.size() << endl;
        }

        /* initialize more variables */
        T_arr = vector<int unsigned>(singleton_count, 0);
        D_arr = vector<int unsigned>(singleton_count, 0);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Initial setup phase: " << duration.count() << " ms" << endl;
    }

    /**
     * @brief Get the total number of elements that we wish to cover
     *
     * @return unsigned long
     */
    unsigned long get_singleton_counts()
    {
        return singleton_count;
    }

    /**
     * @brief Get the candidate token size
     *
     * @return unsigned long
     */
    unsigned long get_candidate_token_size()
    {
        if (candidate_tokens.size() == 0)
        {
            return substring_to_index.size();
        }
        else
        {
            return candidate_tokens.size();
        }
    }

    /**
     * @brief Calculate scores of a given substring defined by its positions in the array
     *
     * @param places of SubstringPos locations of a substring
     * @param T_arr_ptr Array to track assigned tokens
     * @param D_arr_ptr Array to track duplicates
     * @param id_to_count word to count mapping
     * @return long unsigned : score of a particular substring
     */
    long unsigned calculate_score(
        const vector<SubstringPos> &places,
        const vector<int unsigned> *T_arr_ptr,
        const vector<int unsigned> *D_arr_ptr,
        const unordered_map<long unsigned, long unsigned> &id_to_count)
    {
        long unsigned counts = 0;

        unordered_map<long unsigned, vector<SubstringPos>> pplaces;
        for (auto p : places)
        {
            if (pplaces.find(p.arr_start) == pplaces.end())
            {
                pplaces[p.arr_start] = vector<SubstringPos>();
            }
            pplaces[p.arr_start].emplace_back(p);
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

    /**
     * @brief Change graph to reflect new state
     *
     * @param items Substring Positions to cover with rank_idx
     * @param T_arr_ptr Array to track assigned tokens
     * @param D_arr_ptr Array to track duplicates
     * @param rank_idx Assigning elements to tokens with rank
     * @return unordered_set<long unsigned> word start positions affected by change
     */
    unordered_set<long unsigned> alter_graph(
        const vector<SubstringPos> &items,
        vector<int unsigned> *T_arr_ptr,
        vector<int unsigned> *D_arr_ptr,
        const int &rank_idx)
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
                (*T_arr_ptr)[k] = rank_idx;
                (*D_arr_ptr)[k] = d_counter;
            }
            prev_w_start = ws;
        }
        return visited;
    }

    vector<py::bytes> get_ranks()
    {
        vector<py::bytes> pybytes_ranks(0);
        pybytes_ranks.reserve(ranks.size());
        for (auto &r : ranks)
        {
            pybytes_ranks.emplace_back(r);
        }
        return pybytes_ranks;
    }

    /**
     * @brief Advancing the current state with specific tokens
     *
     * @param tokens the order of tokens to be used
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
    pair<vector<py::bytes>, vector<long unsigned>> custom_steps(vector<string> &tokens)
    {
        for (string &token : tokens)
        {
            unsigned int rank = ranks.size();
            ranks.emplace_back(token);
            unsigned long score = calculate_score(substring_to_index[token], &T_arr, &D_arr, id_to_count);
            scores.emplace_back(score);
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
        return pair(get_ranks(), scores);
    }

    /**
     * @brief Advance the current state till we have k number of tokens
     *
     * @param k target number of tokens
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
    pair<vector<py::bytes>, vector<long unsigned>> solve_to_step(unsigned int k)
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
            tbb::parallel_for(
                tbb::blocked_range<long unsigned>(0, items.size()),
                [&](tbb::blocked_range<long unsigned> r)
                {
                    for (long unsigned i = r.begin(); i < r.end(); ++i)
                    {
                        results[items[i]] = calculate_score(
                            substring_to_index[items[i]],
                            &T_arr,
                            &D_arr,
                            id_to_count);
                    }
                });

            pair<string, long unsigned> best = *max_element(results.begin(), results.end(), [](const pair<string, long unsigned> a, const pair<string, long unsigned> b)
                                                            { return a.second < b.second; });
            ranks.emplace_back(best.first);
            scores.emplace_back(best.second);

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
        return pair(get_ranks(), scores);
    }
};

class PyGreedyPCOTokenizer : public GreedyPCOTokenizer
{
public:
    using GreedyPCOTokenizer::build_counter_from_text;
    using GreedyPCOTokenizer::calculate_score;
    using GreedyPCOTokenizer::custom_steps;
    using GreedyPCOTokenizer::get_candidate_token_size;
    using GreedyPCOTokenizer::get_ranks;
    using GreedyPCOTokenizer::get_singleton_counts;
    using GreedyPCOTokenizer::initialize_graph;
    using GreedyPCOTokenizer::solve_to_step;
};

GreedyPCOTokenizer *build(
    unordered_map<string, long unsigned> word_counts = {},
    unordered_set<string> candidate_tokens = {})
{
    return new GreedyPCOTokenizer(word_counts, candidate_tokens);
}

PYBIND11_MODULE(pco_tokenizer, var)
{
    var.doc() = "greedy module";
    py::class_<GreedyPCOTokenizer, PyGreedyPCOTokenizer>(var, "GreedyPCOTokenizer")
        .def(py::init<>(
            [](
                unordered_map<string, long unsigned> word_counts = {},
                unordered_set<string> candidate_tokens = {})
            {
                return new GreedyPCOTokenizer(
                    word_counts,
                    candidate_tokens);
            }))
        .def("get_ranks", &GreedyPCOTokenizer::get_ranks)
        .def("solve_to_step", &GreedyPCOTokenizer::solve_to_step)
        .def("calculate_score", &GreedyPCOTokenizer::calculate_score)
        .def("initialize_graph", &GreedyPCOTokenizer::initialize_graph)
        .def("alter_graph", &GreedyPCOTokenizer::alter_graph)
        .def("custom_steps", &GreedyPCOTokenizer::custom_steps)
        .def("build_counter_from_text", &GreedyPCOTokenizer::build_counter_from_text)
        .def("get_singleton_counts", &GreedyPCOTokenizer::get_singleton_counts)
        .def("get_candidate_token_size", &GreedyPCOTokenizer::get_candidate_token_size);
    var.def("build",
            &build,
            py::arg("word_counts") = unordered_map<string, long unsigned>(),
            py::arg("candidate_tokens") = unordered_set<string>(),
            "Factory function for greedy PCO tokenizer, use this to create your token sets.");
}