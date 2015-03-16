class DataLayer : public BaseLayer {
public:
  string layer_name = "DataLayer";
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
    prev_layer->influence = MatrixXf::Zero(prev_layer->x, prev_layer->y);
    data.maxCoeff(&max_row, &max_col);
    float ans_exp = prev_layer->data(max_row, max_col);
    prev_layer->influence(max_row, max_col) = - learning_rate + learning_rate * ans_exp / prev_layer->data.array().exp().sum();
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
  float delta = 0.001; //安定化のため
};
