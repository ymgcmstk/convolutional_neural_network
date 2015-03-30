class ConvLayer : public BaseLayer {
public:
  ConvLayer () {
    layer_name = "ConvLayer";
  }
  int kernel_x, kernel_y, stride, w;
  int k_half, k_center;
  // paddingがtrueだと入力と出力のxとyが同じになる、alexnetみたいな感じ
  bool padding;
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
    MatrixXf temp_mat, to_slide;
    if (next_layer == NULL) return false;
    if (kernel_x != kernel_y) {
      cout << "conv_layer 17" << endl;
      exit(1);
      //あとで直そう
    }
    data = MatrixXf::Zero(prev_layer->x * prev_layer->y, z);
    for (int i = 0; i < kernel_y; i++) {
      for (int j = 0; j < kernel_x; j++) {
        temp_mat = weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z).transpose();
        to_slide = prev_layer->data * temp_mat;
        data += slide(to_slide, - j + k_half, - i + k_half, prev_layer->y);
      }
    }
    for (int i = 0; i < x * y; i++) data.block(i, 0, 1, z) += bias;
    debug();
    return true;
  }

  bool backward () {
    MatrixXf temp_mat, to_slide, this_influence;
    if (prev_layer == NULL) return false;
    prev_layer->influence = MatrixXf::Zero(prev_layer->x * prev_layer->y, prev_layer->z);
    if (padding) this_influence = influence;
    else this_influence = expand(influence, k_half, y);
    for (int i = 0; i < kernel_y; i++) {
      for (int j = 0; j < kernel_x; j++) {
        //diff計算
        //  i, j番目のおもみをtemp_matに代入
        temp_mat = weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z);
        //  prev_layer->influenceにたす用の行列を計算
        to_slide = this_influence * temp_mat;
        //  slideさせて足す
        prev_layer->influence += slide(to_slide, j - k_half, i - k_half, prev_layer->y);
        //重みの更新
        temp_mat = slide(this_influence, j - k_half, i - k_half, prev_layer->y).transpose();
        weights.block(0, prev_layer->z * (j + i * kernel_y), z, prev_layer->z) -= temp_mat * prev_layer->data;
      }
    }
    //bias
    bias -= MatrixXf::Ones(1, x * y) * influence;
    debug();
    return true;
  }

private:
  //スライドさせる
  MatrixXf slide(MatrixXf target, int slide_x, int slide_y, int this_y) {
    MatrixXf output = MatrixXf::Zero(target.rows(), target.cols());
    for (int i = 0; i < this_y; i++) {
      if (i < -slide_y || i >= this_y - slide_y) {
        continue;
      }
      output.block((i + slide_y) * this_y + max(0, slide_x), 0, this_y - abs(slide_x), target.cols()) =
        target.block(i * this_y - min(0, slide_x), 0, this_y - abs(slide_x), target.cols());
    }
    return output;
  }

  //上下左右をcutだけ抜き出したものを返す
  MatrixXf reduct(MatrixXf target, int cut, int this_y) {
    int row = (this_y - 2 * cut) * (target.rows() / this_y  - 2 * cut);
    int col = target.cols();
    MatrixXf output = MatrixXf::Zero(row, col);
    int new_y = this_y - 2 * cut;
    for (int i = 0; i < this_y; i++) {
      if (i < cut || i >= this_y - cut) {
        continue;
      }
      output.block((i - cut) * new_y, 0, new_y, target.cols()) =
        target.block(i * this_y + cut, 0, new_y, target.cols());
    }
    return output;
  }

  //上下左右にaddだけ追加したものを返す
  MatrixXf expand(MatrixXf target, int cut, int this_y) {
    int row = (this_y + 2 * cut) * (target.rows() / this_y  + 2 * cut);
    int col = target.cols();
    MatrixXf output = MatrixXf::Zero(row, col);
    int new_y = this_y + 2 * cut;
    for (int i = 0; i < this_y; i++) {
      output.block((i + cut) * new_y + cut, 0, this_y, target.cols()) =
        target.block(i * this_y , 0, this_y, target.cols());
    }
    return output;
  }
};
