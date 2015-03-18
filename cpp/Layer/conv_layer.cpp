class ConvLayer : public BaseLayer {
public:
  ConvLayer () {
    layer_name = "ConvLayer";
  }
  int kernel_x, kernel_y, stride, w, pad_x, pad_y;
  int k_half, k_center;
  MatrixXf weights, bias;
  float dropout;//たぶん使わない
  void initialize() {
    k_half = kernel_x / 2;
    k_center = (kernel_x - 1) / 2;
    w = kernel_x * kernel_y * prev_layer->z;
    weights = MatrixXf::Random(z, w);
    /*
      weightsの扱いについて
      weights.block(0, prev_layer->z * (i + j * stride), z, prev_layer->z)がxがi, yがjのz * wというサイズの行列を表現するとする
     */
    bias = MatrixXf::Random(1, z);
  }
  bool forward () {
    if (next_layer == NULL) return false;
    if (kernel_x != kernel_y) {
      cout << "conv_layer 17" << endl;
      exit(1);
      //あとで直そう
    }
    data = MatrixXf::Zero(x * y, z);
    for (int i = 0; i < kernel_y; i++) {
      for (int j = 0; j < kernel_x; j++) {
        temp_mat = weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z).transpose();
        to_slide = prev_layer->data * temp_mat;
        data += slide(to_slide, - j + k_half, - i + k_half, y);
      }
    }
    //bias
    for (int i = 0; i < x * y; i++) data.block(i, 0, 1, z) += bias;
    debug();
    return true;
  }

  bool backward () {
    if (prev_layer == NULL) return false;
    prev_layer->influence = MatrixXf::Zero(prev_layer->x * prev_layer->y, prev_layer->z);
    for (int i = 0; i < kernel_y; i++) {
      for (int j = 0; j < kernel_x; j++) {
        //diff計算
        temp_mat = weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z);
        to_slide = influence * temp_mat;
        prev_layer->influence += slide(to_slide, j - k_half, i - k_half, y);

        //重みの更新
        temp_mat = slide(influence, j - k_half, i - k_half, y).transpose();
        weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z) -= temp_mat * prev_layer->data;
      }
    }
    //bias
    bias -= MatrixXf::Ones(1, x * y) * influence;
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
