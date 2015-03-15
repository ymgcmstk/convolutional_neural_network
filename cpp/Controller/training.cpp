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
  string fname = "../../model_files/mnist_mlp.txt";
  BaseStructure bs;
  init_model_and_data(fname, bs);
  srand((unsigned)time(NULL));
  while (true) {
    unsigned long ind = get_random_long(mnist_max);
    bs.input_layer->data = access_image(ind);
    bs.output_layer->data = access_label(ind);
    bs.forward();
    bs.print_output();
    break;
  }
  return 0;
}
