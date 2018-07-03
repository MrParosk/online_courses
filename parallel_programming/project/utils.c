#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

float* load_data(char* file_name, unsigned int num_samples, unsigned num_features){
    FILE* file_ptr;
    float value;
    float* array = (float*) malloc((num_samples * num_features) * sizeof(float));

    file_ptr = fopen(file_name, "r");

    for(unsigned int i = 0; i < num_samples; i++){
        for (unsigned int j = 0 ; j < num_features; j++){
            fscanf(file_ptr, "%f", &value);
            array[j + i * num_features] = value;
        }
    }

    fclose(file_ptr);
    return array;
}

float uniform(){
    //Uniform [0, 1]
    int max = 10000000;
    float random_value = ((float) (rand() % max)) / ( (float) max);
    return random_value;
}
