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
// #include <boost/regex.hpp>
#include <regex>
#include "tbb.h"
using namespace std;
// using namespace boost;
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

    /**
     * @brief Construct a new Greedy P C O Tokenizer object
     * 
     * @param word_counts word to count mapping
     * @param candidate_tokens to investigate
     */
    GreedyPCOTokenizer(unordered_map<string, long unsigned> &word_counts, unordered_set<string> &candidate_tokens)
        : word_counts(word_counts), candidate_tokens(candidate_tokens)
    {
    }

    virtual ~GreedyPCOTokenizer() {}

    /**
     * @brief Create a bipartite graph representation and allocate spaces for tracking arrays
     */
    void initialize_graph()
    {
        cout << "Word counts size: " << word_counts.size() << endl;
        cout << "Token set size: " << candidate_tokens.size() << endl;
        if (candidate_tokens.size() == 0)
        {
            cout << "Empty token set size selected -> all possible substrings..." << endl;
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
    long unsigned calculate_score(const vector<SubstringPos> &places, const vector<int unsigned> *T_arr_ptr, const vector<int unsigned> *D_arr_ptr, const unordered_map<long unsigned, long unsigned> &id_to_count)
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
    unordered_set<long unsigned> alter_graph(const vector<SubstringPos> &items, vector<int unsigned> *T_arr_ptr, vector<int unsigned> *D_arr_ptr, const int &rank_idx)
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

    /**
     * @brief Advancing the current state with specific tokens
     * 
     * @param tokens the order of tokens to be used
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
    pair<vector<string>, vector<long unsigned>> custom_steps(vector<string> &tokens)
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
        return pair(ranks, scores);
    }

    /**
     * @brief Advance the current state till we have k number of tokens
     * 
     * @param k target number of tokens
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
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
                    results[items[i]] = calculate_score(substring_to_index[items[i]], &T_arr, &D_arr, id_to_count); } });

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
        return pair(ranks, scores);
    }
};

struct TrieNode
{
public:
    unordered_map<char, TrieNode *> to;
    unsigned int rank = 0;
    unsigned int level;
    char c = -1;
    TrieNode()
    {
        to = unordered_map<char, TrieNode *>{};
        level = -1;
    }

    /**
     * @brief Construct a new Trie Node object, meant for internal use
     * 
     * @param current_char 
     * @param current_level 
     */
    TrieNode(const char current_char, const unsigned int current_level)
    {
        to = unordered_map<char, TrieNode *>{};
        c = current_char;
        level = current_level;
    }
    virtual ~TrieNode() {};

    /**
     * @brief Construct a forward-only linked list using Trie Node objects
     * 
     * @param s string of characters/singletons
     * @param start start from index of string
     * @param rank_idx to be assigned to the Trie Node that ends s
     */
    void add_new_sequence(const string &s, const unsigned int &start, const unsigned int &rank_idx)
    {

        if (auto search = to.find(s[start]); search == to.end())
        {
            to[s[start]] = new TrieNode(s[start], level + 1);
        }
        if (start + 1 == s.size())
        { // end of sequence
            to[s[start]]->rank = rank_idx;
        }
        else
        {
            to[s[start]]->add_new_sequence(s, start + 1, rank_idx);
        }
        return;
    }

    /**
     * @brief Determines if there is a Trie Node in c direction
     * 
     * @param c direction to traverse
     * @return true if there is no Trie Node
     * @return false otherwise
     */
    bool is_new_path(const char &c)
    {
        return to.find(c) == to.end();
    }

    /**
     * @brief Link a new Trie Node to current Trie Node
     * 
     * @param t new Trie Node
     */
    void add_new_path(TrieNode &t)
    {
        to[t.c] = &t;
    }

    /**
     * @brief Get the next Trie Node in c direction
     * 
     * @param c 
     * @return TrieNode* 
     */
    TrieNode *get_next_path(const char &c)
    {
        return to[c];
    }
};

struct CoverPos
{
    unsigned int start;
    unsigned int offset;
    unsigned int rank;
    CoverPos() {};
    /**
     * @brief Construct a new Cover Pos object, meant for internal use
     * 
     * @param start_idx of the cover in text
     * @param num_char that separates the start and end indices
     * @param substr_rank rank assigned to this cover
     */
    CoverPos(const unsigned int start_idx, const unsigned int num_char, const unsigned int substr_rank)
    {
        start = start_idx;
        offset = num_char;
        rank = substr_rank;
    }
};

class TrieCache
{
public:
    TrieNode root{};
    vector<string> rules;
    TrieCache() {};
    /**
     * @brief Construct a new Trie Cache object, meant for internal use
     * We use a Trie Cache to interate over text as std::regex is slow.
     * This helps us to determine if there is a possible cover in O(N)
     * 
     * @param rules order of tokens in decreasing cover priority
     */
    TrieCache(vector<string> &rules) : rules(rules)
    {
        for (unsigned int i = 0; i < rules.size(); ++i)
        {
            string rule = rules.at(i);
            if (rule.size() <= 1)
            {
                continue;
            }
            root.add_new_sequence(rules.at(i), 0, i + 1);
        }
        cout << "Trie constructed" << endl;
    }
    virtual ~TrieCache() {}

    /**
     * @brief Determine if there is a subsequence that correspond to a token
     * 
     * @param s string to investigate
     * @param start_idx starting from which index
     * @param end_idx ending at which index
     * @param final_location to store results of possible Cover Pos
     * @return unsigned int of which index the traversal stopped at
     */
    unsigned int traverse(const string &s, const unsigned int &start_idx, const unsigned int &end_idx, vector<CoverPos> *final_location)
    {
        TrieNode *prev = &root;
        if (prev->is_new_path(s[start_idx]))
        {
            return start_idx + 1;
        }
        prev = prev->to[s[start_idx]];
        unsigned int curr_idx = start_idx;
        for (; curr_idx < end_idx;)
        {
            curr_idx++;
            if (prev->is_new_path(s[curr_idx]))
            {
                break;
            }
            prev = prev->to[s[curr_idx]];
            if (prev->rank != 0)
            {
                final_location->emplace_back(CoverPos(start_idx, prev->level + 1, prev->rank));
            }
        }
        return curr_idx;
    }
};

class GreedyTokenizer
{
    unordered_map<string, int unsigned> rules_cache;
    unordered_map<char, int unsigned> singleton_cache;
    TrieCache trie_cache;
    regex re;
    sregex_token_iterator end;

    /**
     * @brief custom function to sort CoverPos, meant for internal use
     * 
     * @param lhs 
     * @param rhs 
     * @return true if lhs have a greater cover priority than rhs
     * @return false otherwise
     */
    static bool cover_pos_sorter(CoverPos const &lhs, CoverPos const &rhs)
    {
        if (lhs.rank == rhs.rank)
        {
            return lhs.start < rhs.start;
        }
        return lhs.rank < rhs.rank;
    }

public:
    vector<string> rules;
    unsigned int max_token_size = 0;

    /**
     * @brief Construct a new Greedy Tokenizer object
     * 
     * @param rules_input order of tokens in decreasing cover priority
     */
    GreedyTokenizer(vector<string> rules_input)
    {
        trie_cache = TrieCache(rules_input);
        for (int unsigned i = 0; i < rules_input.size(); ++i)
        {
            rules_cache[rules_input.at(i)] = i + 1;
            rules.emplace_back(rules_input.at(i));
            if (rules_input.at(i).size() > max_token_size)
            {
                max_token_size = rules.at(i).size();
            }
        }
        for (int unsigned i = 0; i < 256; ++i)
        {
            string b({(char)i});
            if (auto search = rules_cache.find(b); search == rules_cache.end())
            {
                rules.emplace_back(b);
                singleton_cache[(char)i] = rules.size();
                rules_cache[b] = rules.size();
            }
            else
            {
                singleton_cache[(char)i] = search->second;
            }
        }
    }
    virtual ~GreedyTokenizer() {}

    /**
     * @brief Get the rules at token position
     * 
     * @param index of token
     * @return vector<uint8_t> representation of the token
     */
    vector<uint8_t> get_rules(int index)
    {
        return vector<uint8_t>(rules.at(index).begin(), rules.at(index).end());
    }

    /**
     * @brief Tokenize one word, O(W^3) complexity 
     * 
     * @param word to be tokenized
     * @return vector<int unsigned> of tokens' ids
     */
    vector<int unsigned> tokenize_word(const string &word)
    {
        if (rules_cache.find(word) != rules_cache.end())
        {
            return vector<int unsigned>{rules_cache.at(word) - 1};
        }
        vector<int unsigned> result;
        vector<int unsigned> T_arr(word.size(), 0);
        vector<int unsigned> D_arr(word.size(), 0);
        vector<CoverPos> to_check;
        for (unsigned int i = 0; i < word.size(); ++i)
        {
            unsigned int max_substr_length = word.size() - i < max_token_size ? word.size() : i + max_token_size;
            for (unsigned int j = i + 2; j < max_substr_length + 1; ++j)
            {
                string substr = word.substr(i, j - i);
                if (auto search = rules_cache.find(substr); search != rules_cache.end())
                {
                    to_check.emplace_back(CoverPos(i, search->first.size(), search->second));
                }
            }
        }
        sort(to_check.begin(), to_check.end(), &cover_pos_sorter);
        unsigned int prev_rank = 0;
        int unsigned d_counter = 0;
        for (CoverPos cover : to_check)
        {
            int unsigned i = cover.start;
            int unsigned j = i + cover.offset;
            if (i > 0 && T_arr[i - 1] != 0 && T_arr[i - 1] == T_arr[i] && D_arr[i - 1] == D_arr[i])
            {
                continue;
            }
            if (j < word.size() && T_arr[j] != 0 && T_arr[j - 1] == T_arr[j] && D_arr[j - 1] == D_arr[j])
            {
                continue;
            }

            d_counter = cover.rank == prev_rank ? d_counter + 1 : 0;
            for (int unsigned k = i; k < j; ++k)
            {
                T_arr[k] = cover.rank;
                D_arr[k] = d_counter;
            }
            prev_rank = cover.rank;
        }

        for (int unsigned i = 0; i < T_arr.size(); ++i)
        {
            if (T_arr[i] == 0)
            {
                result.emplace_back(rules_cache.at(word.substr(i, 1)) - 1);
            }
            else
            {
                result.emplace_back(T_arr[i] - 1);
                i += rules.at(T_arr[i] - 1).size() - 1;
            }
        }
        return result;
    }

    /**
     * @brief Tokenize text that were already pre-split
     * 
     * @param text list of words
     * @return vector<int unsigned> list of tokens' ids
     */
    vector<int unsigned> tokenize_text_in_parts(const vector<string> &text)
    {
        vector<int unsigned> results;
        for (unsigned int i = 0; i < text.size(); ++i)
        {
            vector<int unsigned> w = tokenize_word(text.at(i));
            results.insert(results.end(), w.begin(), w.end());
        }
        return results;
    }

    /**
     * @brief Tokenize text that were already pre-split
     * 
     * @param texts list of lists of words
     * @return vector<vector<int unsigned>> list of lists of tokens' ids
     */
    vector<vector<int unsigned>> batch_tokenize_in_parts(const vector<vector<string>> &texts)
    {
        vector<vector<int unsigned>> results(texts.size(), vector<int unsigned>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, texts.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize_text_in_parts(texts.at(i));} });
        return results;
    }

    /**
     * @brief tokenizes a portion of texts, O(W^2lgW) complexity
     * 
     * @param text 
     * @param start_idx starting at which index
     * @param offset difference in start and end indices
     * @param to_check a list of Cover Pos objects to investigate
     * @param result place to add tokens' ids
     */
    void tokenize_portion(const string &text, const unsigned int &start_idx, const unsigned int &offset, vector<CoverPos> &to_check, vector<int unsigned> *result)
    {
        if (offset == 1)
        {
            result->emplace_back(singleton_cache.at(text[start_idx]) - 1);
            return;
        }
        if (auto search = rules_cache.find(text.substr(start_idx, offset)); search != rules_cache.end())
        {
            result->emplace_back(search->second - 1);
            return;
        }
        vector<int unsigned> T_arr(offset, 0);

        sort(to_check.begin(), to_check.end(), &cover_pos_sorter);
        for (CoverPos cover : to_check)
        {
            int unsigned i = cover.start - start_idx;
            int unsigned i_o = i - 1;
            if (i > 0 && T_arr[i_o] != 0 && T_arr[i_o] == T_arr[i])
            {
                continue;
            }
            int unsigned j = i + cover.offset;
            int unsigned j_o = j - 1;
            if (j < offset && T_arr[j] != 0 && T_arr[j_o] == T_arr[j])
            {
                continue;
            }
            for (int unsigned k = i; k < j; ++k)
            {
                T_arr[k] = cover.rank;
            }
        }

        for (int unsigned i = 0; i < T_arr.size(); ++i)
        {
            if (T_arr[i] == 0)
            {
                result->emplace_back(singleton_cache.at(text[start_idx + i]) - 1);
            }
            else
            {
                result->emplace_back(T_arr[i] - 1);
                i += rules.at(T_arr[i] - 1).size() - 1;
            }
        }
    }

    /**
     * @brief Tokenize whole strings without splitting
     * 
     * @param text 
     * @param token_results place to store tokenization results
     */
    void tokenize_text(const string &text, vector<unsigned int> *token_results)
    {
        for (unsigned int start_idx = 0; start_idx < text.size(); start_idx++)
        {
            vector<CoverPos> intermediate_results;
            unsigned int end_idx = text.size() < start_idx + max_token_size ? text.size() : start_idx + max_token_size;
            unsigned int best_stop_idx = trie_cache.traverse(text, start_idx, end_idx, &intermediate_results);
            for (unsigned int mid_idx = start_idx + 1; mid_idx < best_stop_idx; mid_idx++)
            {
                unsigned int next_stop_idx = trie_cache.traverse(text, mid_idx, mid_idx + max_token_size, &intermediate_results);
                best_stop_idx = best_stop_idx < next_stop_idx ? next_stop_idx : best_stop_idx;
            }
            tokenize_portion(text, start_idx, best_stop_idx - start_idx, intermediate_results, token_results);
            start_idx = best_stop_idx - 1;
        }
    }

    /**
     * @brief batch tokenize whole strings, O(W^2lgW) complexity
     * 
     * @param texts list of strings
     * @return vector<vector<int unsigned>> 
     */
    vector<vector<int unsigned>> batch_tokenize_whole(const vector<string> &texts)
    {
        vector<vector<int unsigned>> results(texts.size());
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, texts.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                                    tokenize_text(texts.at(i), &results.at(i)); } });
        return results;
    }

    /**
     * @brief Set the regex pattern object
     * 
     * @param pattern 
     */
    void set_regex_pattern(string pattern)
    {
        re = regex(pattern, regex::optimize);
    }

    /**
     * @brief Split texts using std::regex, then tokenize texts
     * 
     * @param texts list of strings
     * @return vector<vector<int unsigned>> 
     */
    vector<vector<int unsigned>> batch_split_and_tokenize(vector<string> &texts)
    {
        vector<vector<int unsigned>> results(texts.size(), vector<int unsigned>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, texts.size()),
                                  [&](tbb::blocked_range<int unsigned> r)
                                  {
                                      for (int unsigned i = r.begin(); i < r.end(); ++i)
                                      {
                                          vector<string> text(sregex_token_iterator(texts[i].begin(), texts[i].end(), re), end);
                                          results[i] = tokenize_text_in_parts(text);
                                      }
                                  });
        return results;
    }

    /**
     * @brief Score covers by turn, for evaluation purposes
     * 
     * @param word 
     * @return vector<pair<int unsigned, int unsigned>> when word gets covered
     */
    vector<pair<int unsigned, int unsigned>> score_covers_per_turn(const string &word)
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

            for (int unsigned i = 0; i < word.size() - substr_size + 1; ++i)
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
                result.emplace_back(pair(r - 1, nones + uniqs.size() - 1));
                i += substr_size - 1;
            }
        }
        return result;
    }

    /**
     * @brief Score covers by turn in batches, for evaluation purposes
     * 
     * @param words 
     * @return vector<vector<pair<int unsigned, int unsigned>>> when words gets covered
     */
    vector<vector<pair<int unsigned, int unsigned>>> batch_score_covers_per_turn(vector<string> &words)
    {
        vector<vector<pair<int unsigned, int unsigned>>> results(words.size(), vector<pair<int unsigned, int unsigned>>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = score_covers_per_turn(words[i]); } });
        return results;
    }

    /**
     * @brief Score constrained covers for evaluation purposes
     * 
     * @param word 
     * @return int unsigned number of constrained covers
     */
    int unsigned score_covers(const string &word)
    {
        return tokenize_word(word).size();
    }

    /**
     * @brief Score constrained covers in batches for evaluation purposes
     * 
     * @param words 
     * @return vector<int unsigned> unsigned number of constrained covers for each respective word
     */
    vector<int unsigned> batch_score_covers(vector<string> &words)
    {
        vector<int unsigned> results(words.size(), 0);
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize_word(words[i]).size(); } });
        return results;
    }

    /**
     * @brief Score max covers for evaluation purposes
     * 
     * @param word 
     * @return int unsigned number of covers
     */
    int unsigned score_max_cover(const string &word)
    {
        vector<bool> T_arr(word.size() - 1, 0);
        int unsigned it, substr_size;
        for (int unsigned i = 0; i < rules.size(); ++i)
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


    /**
     * @brief Score max covers in batches for evaluation purposes
     * 
     * @param words 
     * @return vector<int unsigned> unsigned number of covers for each respective word
     */
    vector<int unsigned> batch_score_max_cover(vector<string> &words)
    {
        vector<int unsigned> results(words.size(), 0);
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = score_max_cover(words[i]); } });
        return results;
    }
};

class PyGreedyPCOTokenizer : public GreedyPCOTokenizer
{
public:
    using GreedyPCOTokenizer::calculate_score;
    using GreedyPCOTokenizer::custom_steps;
    using GreedyPCOTokenizer::get_candidate_token_size;
    using GreedyPCOTokenizer::get_singleton_counts;
    using GreedyPCOTokenizer::initialize_graph;
    using GreedyPCOTokenizer::solve_to_step;
};

GreedyPCOTokenizer *build_Greedy_PCO_Tokenizer(unordered_map<string, long unsigned> &word_counts, unordered_set<string> candidate_tokens = {})
{
    return new GreedyPCOTokenizer(word_counts, candidate_tokens);
}

class PyGreedyTokenizer : public GreedyTokenizer
{
public:
    using GreedyTokenizer::batch_score_max_cover;
    using GreedyTokenizer::batch_score_covers;
    using GreedyTokenizer::batch_score_covers_per_turn;
    using GreedyTokenizer::batch_split_and_tokenize;
    using GreedyTokenizer::batch_tokenize_in_parts;
    using GreedyTokenizer::batch_tokenize_whole;
    using GreedyTokenizer::GreedyTokenizer;
    using GreedyTokenizer::score_max_cover;
    using GreedyTokenizer::score_covers;
    using GreedyTokenizer::score_covers_per_turn;
    using GreedyTokenizer::set_regex_pattern;
    using GreedyTokenizer::tokenize_text_in_parts;
    using GreedyTokenizer::tokenize_word;
};

GreedyTokenizer *build_Greedy_Tokenizer(vector<string> rules)
{
    return new GreedyTokenizer(rules);
}

PYBIND11_MODULE(greedy_builder, var)
{
    var.doc() = "greedy module";
    py::class_<GreedyPCOTokenizer, PyGreedyPCOTokenizer>(var, "GreedyPCOTokenizer")
        .def(py::init<>([](unordered_map<string, long unsigned> &word_counts,
                           unordered_set<string> candidate_tokens = {})
                        { return new GreedyPCOTokenizer(word_counts, candidate_tokens); }))
        .def("solve_to_step", &GreedyPCOTokenizer::solve_to_step)
        .def("calculate_score", &GreedyPCOTokenizer::calculate_score)
        .def("initialize_graph", &GreedyPCOTokenizer::initialize_graph)
        .def("alter_graph", &GreedyPCOTokenizer::alter_graph)
        .def("custom_steps", &GreedyPCOTokenizer::custom_steps)
        .def("get_singleton_counts", &GreedyPCOTokenizer::get_singleton_counts)
        .def("get_candidate_token_size", &GreedyPCOTokenizer::get_candidate_token_size);
    var.def("build_greedy_pco_tokenizer", &build_Greedy_PCO_Tokenizer, "Factory function for greedy PCO tokenizer, use this to create your token sets.");

    py::class_<GreedyTokenizer, PyGreedyTokenizer>(var, "GreedyTokenizer")
        .def(py::init<>([](vector<string> &cover_order)
                        { return new GreedyTokenizer(cover_order); }))
        .def("batch_tokenize_in_parts", &GreedyTokenizer::batch_tokenize_in_parts)
        .def("batch_tokenize_whole", &GreedyTokenizer::batch_tokenize_whole)
        .def("get_rules", &GreedyTokenizer::get_rules)
        .def("batch_split_and_tokenize", &GreedyTokenizer::batch_split_and_tokenize)
        .def("score_covers", &GreedyTokenizer::score_covers)
        .def("tokenize_text_in_parts", &GreedyTokenizer::tokenize_text_in_parts)
        .def("tokenize_word", &GreedyTokenizer::tokenize_word)
        .def("batch_score_covers", &GreedyTokenizer::batch_score_covers)
        .def("score_max_cover", &GreedyTokenizer::score_max_cover)
        .def("batch_score_max_cover", &GreedyTokenizer::batch_score_max_cover)
        .def("score_covers_per_turn", &GreedyTokenizer::score_covers_per_turn)
        .def("batch_score_covers_per_turn", &GreedyTokenizer::batch_score_covers_per_turn)
        .def("set_regex_pattern", &GreedyTokenizer::set_regex_pattern);
    var.def("build_greedy_tokenizer", &build_Greedy_Tokenizer, "Factory function for greedy tokenizer, use this to encode text to tokens.");
}
