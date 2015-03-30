class MaxPoolingLayer : public BaseLayer {
public:
  MaxPoolingLayer () {
    layer_name = "MaxPoolingLayer";
  }
  int kernel_x, kernel_y, stride, padding;
  int k_half, k_center;
  float dropout;//たぶん使わない
  void initialize() {
    k_half = kernel_x / 2;
    k_center = (kernel_x - 1) / 2;
  }
  bool forward () {
    if (next_layer == NULL) return false;
    debug();
    return true;
  }

  bool backward () {
    if (prev_layer == NULL) return false;
    debug();
    return true;
  }

private:
  MatrixXf output, temp_mat, to_slide;
  MatrixXf slide(MatrixXf target, int slide_x, int slide_y, int this_y) {
    output = MatrixXf::Zero(target.rows(), target.cols());
    for (int i = 0; i < this_y; i++) {
      if (i < -slide_y || i >= this_y - slide_y) {
        continue;
      }
      output.block((i + slide_y) * this_y + max(0, slide_x), 0, this_y - abs(slide_x), target.cols()) =
        target.block(i * this_y - min(0, slide_x), 0, this_y - abs(slide_x), target.cols());
    }
    return output;
  }
};
