using namespace std;
#include "random_access.cpp"
using namespace Eigen;


int main (int argc, char *argv[]) {
  if (argc == 1) print_image_and_label(get_random_long());
  else print_image_and_label(atoi(argv[1]));
}
