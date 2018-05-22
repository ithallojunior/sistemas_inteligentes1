#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>


/*             SET OF FUNCTIONS                     */ 


//sigmoid 1/(1+exp(-x))
float sigmoid(float x){
  float exp_value;
    exp_value = exp((double) -x);
    return 1./(1. + exp_value);
}
//sigmoid derivative sigmoid(1 - sigmoid)
float derivative(float x){
  float s = sigmoid(x);
  return s * (1.-s);
}
//my random numbers
float myrandom(void){
  srand((unsigned) time(NULL));
  return (rand() - (RAND_MAX * 0.8))/1000000000.;
}
//xor
void xor_printer(void){
  float X[4][2] = {{0.,0.}, {0.,1.},{1.,0.},{1.,1.}}; //DATA
  float y[4][1] = {{0.},{1.},{1.},{0.}};  //DATA
  //general variables 
  int w0_size = 3; 
  int max_iter = 500; 
  int  xy_d0 = sizeof(X)/ sizeof(X[0]);//SIZEOF VECTOR[X][] 
  int x_d1 = sizeof(X[0])/ sizeof(X[0][0]); //SIZEOF VECTOR[][X]
  int y_d1 = sizeof(y[0])/ sizeof(y[0][0]); //SIZEOF 0 VECTOR[O][X]
  // weights 
  float w0[x_d1][w0_size];
  float w1[w0_size][y_d1];
  float sum0[xy_d0][w0_size];
  float h0[xy_d0][w0_size];
  float sum1[xy_d0][y_d1];
  float y_output [xy_d0][y_d1];
  //counters
  int i, j, k, m;
  
  ////////////////////generating matrices ////////////////////
  
  // for w0
  for (i=0;i<x_d1;i++){
    for (j=0;j<w0_size;j++){
      w0[i][j] = myrandom();
    }
  }
  // for w1
  for (i=0;i<w0_size;i++){
    for (j=0;j<y_d1;j++){
      w1[i][j] = myrandom();
    }
  }
  /////////////////// feedforward //////////////////////////
  
  //matrix multiplication 0 and tf
  for (i=0;i<xy_d0;i++ ){
    for (j=0;j<w0_size;j++){
        sum0[i][j] = 0.;
        for (k=0;k<w0_size;k++){
          sum0[i][j] += X[i][k] * w0[k][j]; 
        }
        h0[i][j] = sigmoid(sum0[i][j]) ;
        //printf("%d %d: %f \n", i, j, h0[i][j] );
    }
  }

  //printf("%d %d %d %d\n", xy_d0,x_d1, xy_d0, y_d1);
  //matrix multiplication 1 and tf
  printf("\nFIRST OUTPUT:\n");
  for (i=0;i<xy_d0;i++ ){
    for (j=0;j<y_d1;j++){
        sum1[i][j] = 0.;
        for (k=0;k<y_d1;k++){
          sum1[i][j] += sum0[i][k] * w1[k][j]; 
        }
        y_output[i][j] = sigmoid(sum1[i][j]) ;
        printf("%.1f|%.1f:   %.3f \n", X[i][0], X[i][1], y_output[i][j] );
    }
  }
  //////////////////// backpropagation //////////////////////
  //TODO

  //////////////////// final printer ////////////////////////
  printf("\nDATA:\n");
  printf("X            y\n");
  for (i=0;i<4;i++){
    printf("%.1f|%.1f:   %.2f\n", X[i][0], X[i][1], y[i][0]);
  }
}

int main() {
  int X[2][3];
  int  size0 = sizeof(X)/ sizeof(X[0]); 
  size_t  size1 = sizeof(X[0])/ sizeof(X[0][0]); 
  printf("Sigmoid test: %f \n", sigmoid(2) );
  printf("Sigmoid derivative %f \n", derivative(2) );
  printf("Random number %f \n", myrandom());
  printf("Size 0 of array: %d \n", size0);
  printf("Size 1 of array: %zu \n", size1);
  xor_printer();
}
