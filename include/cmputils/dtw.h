#ifndef LIBCMPUTILS_DTW_H
#define LIBCMPUTILS_DTW_H

static const double INF = 1e250; // Pseudo-infinite number for this code

struct Element {
    double cost = INF;
    double xy = 0.0;
    int rowCount = 0;
    int columnCount = 0;
};

double dtw(const double *a, const double *b, const int size_a, const int size_b, const int warp_window, const int verbose);

#endif //LIBCMPUTILS_DTW_H
