#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "utils.h"

void comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* in_placeholder_vector){
    /*
    Doing the first computation step, i.e. (X * theta - y) 
        - X matrix of size [num_samples, num_features]
        - theta of size [num_features, 1]
        - y of size [num_samples, 1]
        - in_placeholder_vector of size [num_samples, 1]
    */

    for(unsigned int i = 0; i < X->rows; i++){
        float temp_sum = 0.0f;
        for(unsigned int j = 0; j < X->cols; j++){
            temp_sum += X->values[j + i * X->cols] * theta->values[j];
        }
        in_placeholder_vector->values[i] = (temp_sum - y->values[i]);
    }
}

void comp_step_two(matrix* X_transpose, matrix* in_placeholder_vector, matrix* out_placeholder_vector){
    /*
        Doing the second computation step, i.e. X.T * in_placeholder_vector / num_samples
    */

    for(unsigned int i = 0; i < X_transpose->rows; i++){
        float temp_sum = 0.0f;
        for(unsigned int j = 0; j < X_transpose->cols; j++){
            temp_sum += X_transpose->values[j + i * X_transpose->cols] * in_placeholder_vector->values[j];
        }
        out_placeholder_vector->values[i] = temp_sum/(X_transpose->cols);
    }
}

void update_weights(matrix* theta, matrix* in_placeholder_vector, const float learning_rate, const float lambda){
    for(unsigned int i = 0; i < theta->rows; i++){
        theta->values[i] = theta->values[i] - learning_rate * (in_placeholder_vector->values[i] + (lambda * theta->values[i])/(2.0f));
    }
}

void gradient_descent(matrix* X, matrix* y, matrix* theta, const unsigned int num_iter, const float learning_rate, const float lambda){
    const unsigned int num_samples = X->rows;
    const unsigned int num_features = X->cols;

    matrix* X_transpose = create_matrix(num_samples, num_features);
    equal_matrix(X_transpose, X);
    transpose_matrix(X_transpose);

    matrix* placeholder_1 = create_matrix(num_samples, 1);
    matrix* placeholder_2 = create_matrix(num_features, 1);

    for(unsigned int i = 0; i < num_iter; i++){
        comp_step_one(X, theta, y, placeholder_1);
        comp_step_two(X_transpose, placeholder_1, placeholder_2);
        update_weights(theta, placeholder_2, learning_rate, lambda);
    }

    free_matrix(X_transpose);
    free_matrix(placeholder_1);
    free_matrix(placeholder_2);
}

int main(){
    const unsigned int num_samples = 1000;
    const unsigned int num_features = 5;

    const unsigned int num_iter = 200;
    const float learning_rate = 0.05f;
    const float lambda = 1e-3f;

    float* X_values = load_data("X.txt", num_samples, num_features);
    matrix* X = create_matrix(num_samples, num_features);
    fill_matrix_values(X, X_values);
    free(X_values);

    float* y_values = load_data("y.txt", num_samples, 1);
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
