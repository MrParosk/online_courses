#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "matrix.h"
#define MAX 10000000

float* load_data(char* file_name, const unsigned int num_samples, const unsigned num_features){
    FILE* file_ptr;
    float value;
    float* array = (float*) malloc((num_samples * num_features) * sizeof(float));

    file_ptr = fopen(file_name, "r");

    for(unsigned int i = 0; i < num_samples; i++){
        for(unsigned int j = 0 ; j < num_features; j++){
            fscanf(file_ptr, "%f", &value);
            array[j + i * num_features] = value;
        }
    }

    fclose(file_ptr);
    return array;
}

float uniform(){
    //Uniform [0, 1]
    float random_value = ((float) (rand() % MAX)) / ( (float) MAX);
    return random_value;
}

float rmse(matrix* X, matrix* theta, matrix* y){
    float rmse_value = 0.0f;

    for(unsigned int i = 0; i < X->rows; i++){
        float temp_sum = 0.0f;
        for(unsigned int j = 0; j < X->cols; j++){
            temp_sum += X->values[j + i*X->cols] * theta->values[j];
        }
        rmse_value += (y->values[i] - temp_sum)*(y->values[i] - temp_sum);
    }

    rmse_value = sqrt(rmse_value)/X->rows;
    return rmse_value;
}
