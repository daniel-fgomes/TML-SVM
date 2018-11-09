/* 
 * File:   main.cpp
 * Author: Daniel de Filgueiras Gomes ( daniel.fgomes@ufpe.br )
 * This source code is licensed under the MLP2 terms (MLP2.html or https://www.mozilla.org/en-US/MPL/2.0/ )
 * This code is provided as an illustrative example in Machine Learning course in Department of Electronics and Systems/UFPE.
 * ( https://www.ufpe.br/des/o-des )
 * Created on 17 de Agosto de 2018, 11:00
 */

#include <stdio.h>
#include <stdlib.h>
#include <libsvm/svm.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int main(int argc, char **argv)
{
    
    
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Important warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //This program aims to solve the XOR classification problem.
    //This is an toy problem and for simplicity the cross-validation procedures and parameter settings are not treated here.
    
    struct svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;


    struct svm_problem prob;
    prob.l = 4;//Number of training samples, each sample is a 2d vector.

    //x values matrix of xor values
    double matrix[prob.l][2];
    matrix[0][0] = 1;//sample 1 (2d line vector), label -1(defined in the line 84)
    matrix[0][1] = 1;

    matrix[1][0] = 1;//sample 2(2d line vector), label 1(defined in the line 85)
    matrix[1][1] = -1;

    matrix[2][0] = -1;//sample 3(2d line vector), label 1(defined in the line 86)
    matrix[2][1] = 1;

    matrix[3][0] = -1;//sample 4(2d line vector), label -1(defined in the line 87)
    matrix[3][1] = -1;

    
    svm_node** x = Malloc(svm_node*,prob.l);//<-- vector for a vector

    //Trying to assign from matrix to svm_node training examples
    for (int row = 0;row <prob.l; row++){
        svm_node* x_space = Malloc(svm_node,3);
        for (int col = 0;col < 2;col++){
            x_space[col].index = col;
            x_space[col].value = matrix[row][col];
        }
        x_space[2].index = -1;      //Each row of properties should be terminated with a -1 according to the readme
        x[row] = x_space;
    }

    prob.x = x;

    //yvalues
    prob.y = Malloc(double,prob.l); //<--vector with labels for each sample in the line 54,57...
    prob.y[0] = -1;
    prob.y[1] = 1;
    prob.y[2] = 1;
    prob.y[3] = -1;

    //Train model---------------------------------------------------------------------
    svm_model *model = svm_train(&prob,&param);


    //Test model----------------------------------------------------------------------
    svm_node* testnode1 = Malloc(svm_node,3);//<-- this is one single sample vector (1,0.5)
    testnode1[0].index = 0;
    testnode1[0].value = 1;
    testnode1[1].index = 1;
    testnode1[1].value = 0.5;
    testnode1[2].index = -1;//End of dataset

    //This works correctly:
    // 1 for vectors  (1,-1) or (-1,1) 
    // -1 for vectors (1,1) or (-1,-1)
    double retval = svm_predict(model,testnode1);
    printf("Test 1 retval: %f\n",retval);

    //Now, let's try another vector  , ex (-0.1,1)
    testnode1[0].index = 0;
    testnode1[0].value = -0.1;
    testnode1[1].index = 1;
    testnode1[1].value = 1;
    testnode1[2].index = -1;//End of dataset

    retval = svm_predict(model,testnode1);
    printf("Test 2 retval: %f\n",retval);

    svm_destroy_param(&param);
    
    //Release the allocated memory
    for (int row = 0;row <prob.l; row++){
        free(prob.x[row]);
    }
    
    free(prob.x);
    free(prob.y);
    

    return 0;
}

