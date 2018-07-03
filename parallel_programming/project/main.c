#include <stdio.h>
#include <stdlib.h>
#include "src/matrix.h"
#include "src/utils.h"
#include "src/cpu_gradient_descent.h"

int main(){
    const unsigned int num_samples = 1000;
    const unsigned int num_features = 5;

    const unsigned int num_iter = 200;
    const float learning_rate = 0.05f;
    const float lambda = 1e-3f;

    float* X_values = load_data("data/X.txt", num_samples, num_features);
    matrix* X = create_matrix(num_samples, num_features);
    fill_matrix_values(X, X_values);
    free(X_values);

    float* y_values = load_data("data/y.txt", num_samples, 1);
    matrix* y = create_matrix(num_samples, 1);
    fill_matrix_values(y, y_values);
    free(y_values);

    matrix* theta = create_matrix(num_features, 1);
    fill_matrix_random(theta);

    // True theta = [1.5, 2.2, 3.5, -2.3, -0.5]
    gradient_descent(X, y, theta, num_iter, learning_rate, lambda);
    printf("Theta values: \n");
    print_matrix(theta);
    float rmse_value = rmse(X, theta, y);
    printf("\n");
    printf("RMSE value: %f", rmse_value);

    free_matrix(X);
    free_matrix(y);
    free_matrix(theta);

    return 0;
}
