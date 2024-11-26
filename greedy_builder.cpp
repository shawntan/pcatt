#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <vector>
#include <string>
#include <numeric>
#include "oneapi/tbb.h"

using namespace std;

// c++ -O3 -Wall -shared -std=c++20 -ltbb -fPIC $(python3 -m pybind11 --includes) greedy_builder.cpp -o greedy_builder$(python3-config --extension-suffix)

class Greedy_Builder
{
    const vector<string> rules;

public:
    Greedy_Builder(vector<string> &rules)
        : rules(rules)
    {
    }
    virtual ~Greedy_Builder() {}

    vector<int unsigned> tokenize(const string &word)
    {
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
                result.push_back((int unsigned)word[i]);
            }
            else
            {
                result.push_back(256 + T_arr[i]);
                i += rules.at(T_arr[i] - 1).size() - 1;
            }
        }
        return result;
    }

    vector<vector<int unsigned>> batch_tokenize(vector<string> &words)
    {
        vector<vector<int unsigned>> results(words.size(), vector<int unsigned>{});
        oneapi::tbb::parallel_for(tbb::blocked_range<int unsigned>(0, words.size()), [&](tbb::blocked_range<int unsigned> r)
                                  { for (int unsigned i=r.begin(); i<r.end(); ++i){
                    results[i] = tokenize(words[i]); } });
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

class PyGreedy_Builder : public Greedy_Builder
{
public:
    using Greedy_Builder::batch_score_max_cover;
    using Greedy_Builder::batch_score_max_merges;
    using Greedy_Builder::batch_tokenize;
    using Greedy_Builder::Greedy_Builder;
    using Greedy_Builder::score_max_cover;
    using Greedy_Builder::score_max_merges;
    using Greedy_Builder::tokenize;

    // vector<long unsigned> tokenize(string &word) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         vector<long unsigned>,
    //         Greedy_Builder,
    //         tokenize,
    //         word
    //     );
    // }

    // vector<vector<long unsigned>> batch_tokenize(vector<string> &words) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         vector<vector<long unsigned>>,
    //         Greedy_Builder,
    //         batch_tokenize,
    //         words
    //     );
    // }

    // long unsigned score_max_merges(vector<string> &word) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         long unsigned,
    //         Greedy_Builder,
    //         score_max_merges,
    //         word
    //     );
    // }

    // vector<long unsigned> batch_score_max_merges(vector<string> &words) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         vector<long unsigned>,
    //         Greedy_Builder,
    //         batch_score_max_merges,
    //         words
    //     );
    // }

    // long unsigned score_max_cover(vector<string> &word) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         long unsigned,
    //         Greedy_Builder,
    //         score_max_cover,
    //         word
    //     );
    // }

    // vector<long unsigned> batch_score_max_cover(vector<string> &words) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         vector<long unsigned>,
    //         Greedy_Builder,
    //         batch_score_max_cover,
    //         words
    //     );
    // }
};

Greedy_Builder *build(vector<string> rules)
{
    return new Greedy_Builder(rules);
}

PYBIND11_MODULE(greedy_builder, var)
{
    var.doc() = "greedy module";
    py::class_<Greedy_Builder, PyGreedy_Builder>(var, "Greedy_Builder")
        // .def(py::init<vector<string>>())
        .def(py::init<>([](vector<string> &cover_order)
                        { return new Greedy_Builder(cover_order); }))
        .def("tokenize", &Greedy_Builder::tokenize)
        .def("batch_tokenize", &Greedy_Builder::batch_tokenize)
        .def("score_max_merges", &Greedy_Builder::score_max_merges)
        .def("batch_score_max_merges", &Greedy_Builder::batch_score_max_merges)
        .def("score_max_cover", &Greedy_Builder::score_max_cover)
        .def("batch_score_max_cover", &Greedy_Builder::batch_score_max_cover);
    var.def("build", &build, "factory function");
}
