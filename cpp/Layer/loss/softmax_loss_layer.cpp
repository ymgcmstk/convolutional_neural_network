class SoftmaxLossLayer : public BaseLayer {
public:
  SoftmaxLossLayer () {
    layer_name = "SoftmaxLossLayer";
  }
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    debug();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    //cross entropy、なぜかうまくいかない。
    //影響の計算が違うのだろうか
    //そもそも間違ってる,やり直し
    data.maxCoeff(&max_row, &max_col);
    float ans_exp = prev_layer->data(max_row, max_col);
    prev_layer->influence = prev_layer->data.array().exp();
    prev_layer->influence = prev_layer->influence / prev_layer->influence.sum();
    prev_layer->influence(max_row, max_col) -= 1;
    prev_layer->influence *= learning_rate;
    debug();
    return true;
  }
  void white () {
    //whitening用のdataが必要
  }
  bool is_correct () {
    data.maxCoeff(&max_row, &max_col);
    int ans_ind = max_row;
    prev_layer->data.maxCoeff(&max_row, &max_col);
    return ans_ind == (int)max_row;
  }
private:
  MatrixXf::Index max_row, max_col;
};
