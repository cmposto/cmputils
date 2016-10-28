#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include "../include/cmputils/dtw.h"
#include "vector_a.h"
#include "vector_b.h"

#include <stdio.h>

using std::vector;
using namespace testing;

TEST (DTWTests, CheckCost) {
    vector<double> a = {0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0};
    vector<double> b = {1.0, 1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 0.0};
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 3, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(distance, Eq(3));
}

TEST (DTWTests, CheckCostLarge) {
    vector<double> a = a_400;
    vector<double> b = b_400;
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 15, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(std::round(distance), Eq(14379) );
}

TEST (DTWTests, CheckCostLargeInt100) {
    vector<double> a = a_100;
    vector<double> b = b_100;
    for(double &v: a){v = std::round(v);}
    for(double &v: b){v = std::round(v);}
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 5, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(distance, Eq(691) );
}

TEST (DTWTests, CheckCostLargeInt200) {
    vector<double> a = a_200;
    vector<double> b = b_200;
    for(double &v: a){v = std::round(v);}
    for(double &v: b){v = std::round(v);}
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 10, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(distance, Eq(7149) );
}

TEST (DTWTests, CheckCostLargeInt400) {
    vector<double> a = a_400;
    vector<double> b = b_400;
    for(double &v: a){v = std::round(v);}
    for(double &v: b){v = std::round(v);}
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 20, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(std::round(distance), Eq(14369) );
}

TEST (DTWTests, CheckCostLargeInt2000) {
    vector<double> a = a_400;
    for(int i=0; i<4; i++) for(double val: a_400) a.push_back(val);
    vector<double> b = b_400;
    for(int i=0; i<4; i++) for(double val: b_400) b.push_back(val);

    for(double &v: a){v = std::round(v);}
    for(double &v: b){v = std::round(v);}
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), 20, 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(std::round(distance), Eq(71377) );
}

TEST (DTWTests, CheckCostLargeIntNoWarp2000) {
    vector<double> a = a_400;
    for(int i=0; i<4; i++) for(double val: a_400) a.push_back(val);
    vector<double> b = b_400;
    for(int i=0; i<4; i++) for(double val: b_400) b.push_back(val);

    for(double &v: a){v = std::round(v);}
    for(double &v: b){v = std::round(v);}
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), a.size(), 1);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(std::round(distance), Eq(71377) );
}


TEST (DTWTests, CheckWarpingWindow) {
    vector<double> a = {0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0};
    vector<double> b = {1.0, 1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 0.0};
    double distance = dtw(a.data(), b.data(), a.size(), b.size(), a.size(), 2);
    printf("Distance: %f \n", distance);
    ASSERT_THAT(distance, Eq(3));
}
