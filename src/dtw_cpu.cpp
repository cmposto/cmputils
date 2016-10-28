#include "../include/cmputils/dtw.h"
#include <iostream>
#include <chrono>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

using us = std::chrono::microseconds;
using get_time = std::chrono::steady_clock;
using std::chrono::duration_cast;

double distance(double a, double b) {
    return std::abs(a-b);
}

//coded for testing purposes ONLY
double dtw(const double *a, const double *b, const int size_a, const int size_b, const int warp_window, const int verbose){
    vector<vector<Element>> costs;

    auto t1 = get_time::now();

    //initialize costs -> update all rows for column 0
    for(size_t a_idx = 0; a_idx<size_a; a_idx++){
        // create the matrix
        vector<Element> costs_x(size_b);
        costs.push_back(costs_x);

        //set first element in each row for first col
        double xy = distance(a[a_idx], b[0]);
        costs[a_idx][0].xy = xy;

        if (a_idx == 0) {
            costs[a_idx][0].cost = xy;
        } else {
            if(warp_window<0) {
                costs[a_idx][0].cost = costs[a_idx - 1][0].cost + xy;
            } else {
                costs[a_idx][0].rowCount = costs[a_idx - 1][0].rowCount + 1;
                if (costs[a_idx][0].rowCount < warp_window) {
                    costs[a_idx][0].cost = costs[a_idx - 1][0].cost + xy;
                }
            }
        }
    }

    //initialize costs -> update all columns for row 0
    for(size_t b_idx = 0; b_idx<size_b; b_idx++){
        //set first element in each row for first col
        double xy = distance(a[0], b[b_idx]);
        costs[0][b_idx].xy = xy;

        if (b_idx == 0) {
            costs[0][b_idx].cost = xy;
        } else {
            if (warp_window<0) {
                costs[0][b_idx].cost = costs[0][b_idx - 1].cost + xy;
            } else {
                costs[0][b_idx].columnCount = costs[0][b_idx - 1].columnCount + 1;
                if (costs[0][b_idx].columnCount < warp_window) {
                    costs[0][b_idx].cost = costs[0][b_idx - 1].cost + xy;
                }
            }
        }
    }

    //compute min costs
    for(long long a_idx = 1; a_idx<size_a; a_idx++){
        //int b_idx_start = max<long>(1, (warp_window<0)?1:a_idx-warp_window);
        //int b_idx_stop = min<long>(size_b, (warp_window<0)?size_b:a_idx+warp_window+1);
        int b_idx_start = 1;
        int b_idx_stop = size_b;
        if(verbose>1) {
            // log index interval
            cout << a_idx << " -> " << b_idx_start << ", " << b_idx_stop << endl;
        }
        for(long long b_idx = b_idx_start; b_idx<b_idx_stop; b_idx++){
            double xy = distance(a[a_idx], b[b_idx]);
            Element &t = costs[a_idx][b_idx];
            Element &tx = costs[a_idx - 1][b_idx - 1];
            Element &tc = costs[a_idx - 1][b_idx];
            Element &tr = costs[a_idx][b_idx - 1];
            t.xy = xy;
            if(warp_window<0) {
                // ignore warp window
                t.cost = min(min(tx.cost, tc.cost), tr.cost) + xy;
            } else {
                if (tx.cost <= tr.cost && tx.cost <= tc.cost) {
                    t.cost = tx.cost + xy;
                } else if (tr.cost < tc.cost && tr.rowCount < warp_window) {
                    t.cost = tr.cost + xy;
                    t.rowCount = tr.rowCount + 1;
                } else if (tc.columnCount < warp_window) {
                    t.cost = tc.cost + xy;
                    t.columnCount = tc.columnCount + 1;
                } else {
                    t.cost = tx.cost + xy;
                }
            }
        }
    }

    if(verbose>1) {
        // print cumulated cost
        for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
            for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                cout << costs[a_idx][b_idx].cost << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    if(verbose>2) {
        // print individual cost
        for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
            for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                cout << costs[a_idx][b_idx].xy << " ";
            }
            cout << endl;
        }
    }

    auto t2 = get_time::now();

    if(verbose>0) {
        cout << " TOTAL=" << duration_cast<us>(t2 - t1).count() << " DIST: " << (costs[size_a-1][size_b-1].cost) << endl;
    }

    return costs[size_a-1][size_b-1].cost;
}
