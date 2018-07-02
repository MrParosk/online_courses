#include <stdio.h>
#include "matrix.h"

void comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* temp_vector){
    /*
    Doing the first computation step, i.e. (X * theta - y)/num_samples 
        - X matrix of size [num_samples, num_features]
        - theta of size [num_features, 1]
        - y of size [num_samples, 1]
        - temp_vector of size [num_samples, 1]
    */

    for(int i=0; i < X->rows; i++){
        float temp_sum = 0.0f;
        for(int j=0; j < X->cols; j++){
            temp_sum += X->values[j + i * X->cols] * theta->values[j];
        }
        temp_vector->values[i] = (temp_sum - y->values[i]);
    }
}

int main(){

    unsigned int num_samples = 5;
    unsigned int num_features = 1;

    float X_values[10] = {1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    matrix* X = create_matrix(num_samples, num_features + 1); // Adding one extra column due to bias term
    fill_matrix_values(X, X_values);

    float y_values[5] = {1.15899391f, 1.99822109f, 2.94081932f, 3.84201378f, 5.1358969f};
    matrix* y = create_matrix(num_samples, 1);
    fill_matrix_values(y, y_values);
    
    matrix* theta = create_matrix(num_features + 1, 1);    
    fill_matrix_random(theta);
    print_matrix(theta);

    return 0;
}
