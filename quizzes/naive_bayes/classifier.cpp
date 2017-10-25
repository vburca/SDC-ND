#define _USE_MATH_DEFINES

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <math.h>
#include <map>
#include <numeric>
#include <vector>
#include "classifier.h"

using namespace std;

typedef shared_ptr<vector<shared_ptr<vector<double>>>> vectmatrix;

template <class T>
void print_vector(const vector<T>* vv)
{
    if (vv->size() == 0)
    {
        cout << "[]" << endl;
        return;
    }

    cout << "[ ";
    for (int i = 0; i < vv->size(); i++)
    {
        cout << vv->at(i) << " ";
    }
    cout << "]" << endl;
}


/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

double GNB::gaussian_probability(double obs, double mu, double sig)
{
    double num = pow(obs - mu, 2.0);
    double denum = pow(2 * sig, 2.0);
    double norm = 1 / sqrt(pow(2 * M_PI * sig, 2.0));

    return norm * exp(-num / denum);
}


void GNB::train(vector<vector<double>> data, vector<string> labels)
{
    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                  [3.5, 0.1, 5.9, -0.02],
                  [8.0, -0.3, 3.0, 2.2],
                  ...
              ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    map<string, vectmatrix> totals_by_label;

    int n_vars = 4;

    for (string label : this->possible_labels)
    {
        vectmatrix totals = make_shared<vector<shared_ptr<vector<double>>>>();

        for (int i = 0; i < n_vars; i++)
        {
            totals->push_back(make_shared<vector<double>>());
        }

        totals_by_label.insert(pair<string, vectmatrix>(label, totals));
    }

    // For each data vector, aggregate each of the 4 data vector vars per
    // label type
    for (int i = 0; i < data.size(); i++)
    {
        vector<double> X = data.at(i);
        string Y = labels.at(i);

        vectmatrix totals = totals_by_label[Y];
        // For each of the 4 var types
        // s, d, s_dot, d_dot
        // Aggregate them in arrays (for each type) within the
        // current label
        for (int j = 0; j < n_vars; j++)
        {
            double var = X.at(j);
            shared_ptr<vector<double>> var_vect = totals->at(j);
            var_vect->push_back(var);
        }
    }

    // Now that we have everything aggregated, we can calculate the mean
    // and std per each var (i.e. s, d, s_dot, d_dot)
    // means: [ [left_s_mean, left_d_mean, ...], [keep_s_mean, keep_d_mean, ...], ...]
    for (string label : this->possible_labels)
    {
        vector<double> means;
        vector<double> stds;

        vectmatrix totals = totals_by_label[label];
        for (int i = 0; i < totals->size(); i++)
        {
            shared_ptr<vector<double>> var_vect = totals->at(i);

            // Calculate mean for this var
            double sum = accumulate(var_vect->begin(), var_vect->end(), 0.0);
            double mean = sum / var_vect->size();
            means.push_back(mean);

            // Calculate stddev for this var
            // declare vector of differences (x[i] - mean) and calculate it
            vector<double> diffs(var_vect->size());
            transform(var_vect->begin(), var_vect->end(), diffs.begin(),
                        [mean](double x) { return x - mean; });
            // calculate the squared sum of the differences
            double sq_sum = inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0);
            double stddev = sqrt(sq_sum / var_vect->size());
            stds.push_back(stddev);
        }

        this->means.push_back(means);
        this->stds.push_back(stds);
    }
}

string GNB::predict(vector<double> sample)
{
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
    */
    vector<double> probs;

    for (int i = 0; i < this->possible_labels.size(); i++)
    {
        vector<double> means = this->means.at(i);
        vector<double> stds = this->stds.at(i);
        string label = this->possible_labels.at(i);

        double product = 1;
        for (int j = 0; j < sample.size(); j++)
        {
            double mu = means.at(j);
            double sig = stds.at(j);
            double var = sample.at(j);

            double likelihood = gaussian_probability(var, mu, sig);
            product *= likelihood;
        }

        probs.push_back(product);
    }

    double sum = accumulate(probs.begin(), probs.end(), 0.0);
    // normalize probs
    transform(probs.begin(), probs.end(), probs.begin(), [sum](double p) { return p / sum; });

    // Find best probability
    int imax = 0;
    double pmax = 0;
    for (int i = 0; i < probs.size(); i++)
    {
        if (probs.at(i) > pmax)
        {
            pmax = probs.at(i);
            imax = i;
        }
    }

    return this->possible_labels[imax];
}