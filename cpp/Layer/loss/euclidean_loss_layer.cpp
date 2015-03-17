class EuclideanLossLayer : public BaseLayer {
public:
  EuclideanLossLayer () {
    layer_name = "EuclideanLossLayer";
  }
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    debug();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    prev_layer->influence = learning_rate * 2 * (prev_layer-> data - data);
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
