#ifndef CPU_GRADIENT_DESCENT_H
#define CPU_GRADIENT_DESCENT_H
#include "matrix.h"

void comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* in_placeholder_vector);

void comp_step_two(matrix* X_transpose, matrix* in_placeholder_vector, matrix* out_placeholder_vector);

void update_weights(matrix* theta, matrix* in_placeholder_vector, const float learning_rate, const float lambda);

void gradient_descent(matrix* X, matrix* y, matrix* theta, const unsigned int num_iter, const float learning_rate, const float lambda);

#endif