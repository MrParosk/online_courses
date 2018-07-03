#ifndef UTILS_H
#define UTILS_H
#include "matrix.h"

float* load_data(char* file_name, const unsigned int num_samples, const unsigned num_features);

float uniform();

float rmse(matrix* X, matrix* theta, matrix* y);

#endif
