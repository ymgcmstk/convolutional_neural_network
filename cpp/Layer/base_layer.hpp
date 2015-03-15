#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class BaseLayer {
public:
  virtual void initialize () {}
  virtual bool forward () {
    return true;
  }
  virtual bool backward () {
    return true;
  }
  int x, y, z;
  MatrixXf data;
  void print() {
    for (int i = 0; i < x * y * z; i++) cout << data(i, 0) << " ";
    cout << endl;
  }
  MatrixXf vectorize () {
    if (data.cols() == 1) return data;
    MatrixXf vectorized_data(x * z * y, 1);
    for (int i = 0; i < z; i++) {
      vectorized_data.block(i * x * z, 0, x * z, 1) = data.block(0, i, x * z, 1);
    }
    return vectorized_data;
  }
  BaseLayer *next_layer, *prev_layer;
};