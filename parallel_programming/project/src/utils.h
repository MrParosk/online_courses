#ifndef UTILS_H
#define UTILS_H
#include "matrix.h"

int file_exists(const char* filename);

float* load_data(const char* file_name, const unsigned int num_samples, const unsigned num_features);

float uniform();

float rmse(matrix* X, matrix* theta, matrix* y);

#endif
