#include <ctime>
#include <cstdlib>
#include <iostream>
#include "../Structure/base_structure.cpp"
#include "../Utility/init_model_and_data.cpp"
#include "../../mnist_data/api/random_access.cpp"

unsigned long mnist_max = 60000;

unsigned long get_random_long(unsigned long n_max) {
  unsigned long temp;
  while (true) {
    temp = (rand() % 2) * RAND_MAX + rand();
    if (temp < n_max) {
      return temp;
    }
  }
}

int main () {
  string fname = "../../model_files/xor_mlp.txt";
  //string fname = "../../model_files/mnist_mlp.txt";
  //string fname = "../../model_files/debug_net.txt";
  BaseStructure bs;
  init_model_and_data(fname, bs);
  srand((unsigned)time(NULL));
  int iter = 0;
  MatrixXf mean = calc_mean(10000);
  MatrixXf::Index max_row, max_col;
  float var = calc_var_float(10000, mean);
  int correct_num = 0;
  int term = 10;
  int rand_a, rand_b;
  MatrixXf input_mat(2,1);
  MatrixXf output_mat(2,1);
  while (true) {
    rand_a = rand()%2;
    rand_b = rand()%2;
    unsigned long ind = get_random_long(mnist_max);
    input_mat << rand_a, rand_b;
    bs.input_layer->data = input_mat;
    output_mat << (float)((rand_a + rand_b) == 1), (float)((rand_a + rand_b) != 1);
  bs.output_layer->data = output_mat;
    bs.forward();
    if (bs.is_correct()) correct_num++;
    bs.backward();
    iter++;
    if (iter % term == 0) {
      cout << "iter : " << iter << endl;
      cout << "accuracy : " << (float)correct_num / term << endl;
      cout << bs.input_layer->data.transpose() << endl;
      cout << bs.output_layer->data.transpose() << endl;
      bs.print_output();
      correct_num = 0;
      //cout << iter << endl;
    }
  }
  return 0;
}
