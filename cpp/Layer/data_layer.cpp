class DataLayer : public BaseLayer {
public:
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    //とりあえずcross entropy誤差
    prev_layer->influence = MatrixXf::Zero(prev_layer->x, prev_layer->y);
    data.maxCoeff(&max_row, &max_col);
    prev_layer->influence(max_row, max_col) = - learning_rate / prev_layer->data(max_row, max_col);
    return true;
  }
  void white () {
    //whitening用のdataが必要
  }
private:
  MatrixXf::Index max_row, max_col;
};
