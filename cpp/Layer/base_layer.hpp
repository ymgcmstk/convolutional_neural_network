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
  virtual bool is_correct () {
    return false;
  }
  string layer_name;
  int x, y, z;
  MatrixXf data, influence;
  float learning_rate;
  void debug () {
    //hasAnormal(true);
  }
  bool hasNan(bool output) {
    for_checking_anormal = vectorize(true);
    for (int i = 0; i < x * y * z; i++) {
      if (isnan(for_checking_anormal(i, 0))) {
        if (output) cout << "nan is detected in " << layer_name << endl;
        cout << data << endl;
        return true;
      }
    }
    return false;
  }
  bool hasInf(bool output) {
    for_checking_anormal = vectorize(true);
    for (int i = 0; i < x * y * z; i++) {
      if (isinf(for_checking_anormal(i, 0))) {
        if (output) cout << "inf is detected in " << layer_name << endl;
        return true;
      }
    }
    return false;
  }
  bool hasAnormal (bool output) {
    return hasNan(output) || hasInf(output);
  }
  void print() {
    for (int i = 0; i < x * y * z; i++) cout << data(i, 0) << " ";
    cout << endl;
  }
  MatrixXf vectorize (bool horizontal) {
    if (horizontal && data.cols() == 1) return data;
    if ((! horizontal) && data.rows() == 1) return data;
    vectorized = data;
    vectorized.resize(vectorized.cols() * vectorized.rows(), 1);
    if (! horizontal) {
      vectorized.transposeInPlace();
    }
    return vectorized;
  }
  BaseLayer *next_layer, *prev_layer;
private:
  MatrixXf for_checking_anormal, vectorized;
};

/*
memo
各layerの各出力ごとにd(loss) / d(output_ij)を計算しそれを各layerに保持させる
だから次の層で計算して前の層に渡すのが丸い
 */
