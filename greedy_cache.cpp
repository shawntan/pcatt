#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include "oneapi/tbb.h"

using namespace std;
namespace chrono = std::chrono;

// g++ -std=c++20 -o greedy.exe greedy_cache.cpp -ltbb -O3

vector<string> splitString(string &str, const char &splitter)
{
    vector<string> result;
    string current = "";
    for (int i = 0; i < str.size(); i++)
    {
        if (str[i] == splitter)
        {
            if (current != "")
            {
                result.push_back(current);
                current = "";
            }
            continue;
        }
        current += str[i];
    }
    if (current.size() != 0)
        result.push_back(current);
    return result;
}

unordered_map<string, long unsigned> get_counts(const string &domain)
{
    vector<string> keys;
    vector<long unsigned> values;
    unordered_map<string, long unsigned> outputs;
    fstream data_file;

    data_file.open("cpp_inputs/counts/" + domain + ".txt", ios::in);
    if (data_file.is_open())
    {
        string data;
        while (getline(data_file, data))
        {
            values.push_back(stoi(data));
        }
    }
    data_file.close();
    data_file.open("cpp_inputs/words/" + domain + ".txt", ios::in);
    string full_string = "";
    if (data_file.is_open())
    {
        string data;
        while (getline(data_file, data))
        {
            full_string = full_string + data + '\n';
        }
        full_string.pop_back();
        for (string &s : splitString(full_string, ' '))
        {
            keys.push_back(s);
        }
    }

    cout << "keys size " << keys.size() << endl;
    cout << "values size " << values.size() << endl;

    for (int i = keys.size() - 1; i >= 0; --i)
    {
        outputs[keys[i]] = values[i];
    }

    return outputs;
}

struct SubstringPos
{
    long unsigned arr_start;
    long unsigned arr_end;
    int word_start;
    int word_end;
    SubstringPos() {};
    SubstringPos(long unsigned a, long unsigned b, int c, int d)
    {
        arr_start = a;
        arr_end = b;
        word_start = c;
        word_end = d;
    }
};

bool sp_sorter(SubstringPos const &lhs, SubstringPos const &rhs)
{
    return lhs.arr_start < rhs.arr_start;
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
            pp.second.begin(), pp.second.end(), &sp_sorter);

        for (auto &p : pp.second)
        {

            const long unsigned ws = p.arr_start;
            const long unsigned we = p.arr_end;
            const int i = p.word_start;
            const int j = p.word_end;
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
            unordered_set<int> uniqs;
            for (int k = i; k < j; ++k)
            {
                if ((*T_arr_ptr)[ws + k] == 0)
                {
                    nones += 1;
                }
                else
                {
                    uniqs.insert((*T_arr_ptr)[ws + k]);
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
        const int i = p.word_start;
        const int j = p.word_end;

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

int main(int argc, char *argv[])
{
    int thread_count = 10;
    string domain = argv[1];
    int k = stoi(argv[2]);
    auto total_start = chrono::high_resolution_clock::now();
    auto start = chrono::high_resolution_clock::now();
    unordered_map<string, long unsigned> word_counts = get_counts(domain);

    cout << "counts size " << word_counts.size() << endl;

    long unsigned char_count = 0;
    long unsigned next_id = 0;
    long unsigned end_id = 0;
    unordered_map<long unsigned, long unsigned> id_to_count;
    unordered_map<string, pair<long unsigned, long unsigned>>
        word_to_index;
    unordered_map<long unsigned, string> index_to_word;
    unordered_map<string, vector<SubstringPos>> substring_to_index;
    unordered_map<string, unordered_set<string>> word_to_substring;

    for (auto &item : word_counts)
    {

        char_count += item.first.size();

        end_id = next_id + item.first.size();
        id_to_count[next_id] = item.second;

        word_to_index[item.first] = pair(next_id, end_id);
        word_to_substring[item.first] = unordered_set<string>();
        for (int i = 0; i < item.first.size(); ++i)
        {

            for (int j = i + 1; j < item.first.size() + 1; ++j)
            {
                string substr = item.first.substr(i, j - i);
                if (substr.size() <= 1)
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

    vector<int unsigned> T_arr(char_count, 0);
    vector<int unsigned> D_arr(char_count, 0);
    unordered_set<string> shortlist;
    vector<string> saved_merges;
    unordered_map<string, int> ranks;
    vector<long unsigned> scores;
    for (auto &s : substring_to_index)
    {
        shortlist.insert(s.first);
    }
    unordered_map<string, long unsigned> results;
    for (auto &s : shortlist)
    {
        results[s] = 0;
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Initial Setup: "
         << duration.count() << " ms" << endl;

    cout << "starting" << endl;
    for (int rank = 1; rank <= k; ++rank)
    {
        start = chrono::high_resolution_clock::now();

        vector<string> items(shortlist.begin(), shortlist.end());
        oneapi::tbb::parallel_for(tbb::blocked_range<long unsigned>(0, items.size()), [&](tbb::blocked_range<long unsigned> r)
                                  { for (long unsigned i=r.begin(); i<r.end(); ++i){
                    results[items[i]] = get_score_helper(substring_to_index[items[i]], &T_arr, &D_arr, id_to_count); } });

        pair<string, long unsigned> best = *max_element(results.begin(), results.end(), [](const pair<string, long unsigned> a, const pair<string, long unsigned> b)
                                                        { return a.second < b.second; });
        ranks[best.first] = rank;
        scores.push_back(best.second);

        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        unordered_set<long unsigned> visited = alter_graph(substring_to_index[best.first], &T_arr, &D_arr, rank);

        shortlist.clear();
        for (auto &v : visited)
        {
            shortlist.insert(word_to_substring[index_to_word[v]].begin(),
                             word_to_substring[index_to_word[v]].end());
        }
        for (auto &r : ranks)
        {
            shortlist.erase(r.first);
        }
        results.erase(best.first);

        stop = chrono::high_resolution_clock::now();
        auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << rank << ". |" << best.first << " | " << best.second << " | " << duration.count() << " ms | " << duration2.count() << " ms | shortlist: " << shortlist.size() << endl;
    }

    string out_dir = "cpp_outputs/" + domain;
    if (!filesystem::is_directory(out_dir) || !filesystem::exists(out_dir)) {
        filesystem::create_directory(out_dir);
    }
    ofstream f;
    f.open(out_dir + "/tokens.txt");
    for (auto r : ranks)
    {
        f << r.first << " ";
    }
    f.close();
    f.open(out_dir + "/merges.txt");
    for (auto s : scores)
    {
        f << s << endl;
    }
    f.close();
    stop = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(stop - total_start);
    cout << "total time taken: " << total_duration.count() << " seconds" <<endl;
}
