class MaxPoolingLayer : public BaseLayer {
public:
  MaxPoolingLayer () {
    layer_name = "MaxPoolingLayer";
  }
  int kernel_size, stride;
  void initialize() {
    wy = prev_layer->x * prev_layer->y * prev_layer->z;
    weights = MatrixXf::Random(wx, wy);
    bias = MatrixXf::Random(wx, 1);
  }
  bool forward () {
    if (next_layer == NULL) return false;
    data = weights * prev_layer->vectorize(true) + bias;
    debug();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    prev_layer->influence = weights.transpose() * influence;
    weights = weights - influence.asDiagonal() * (MatrixXf::Ones(wx, 1) * prev_layer->vectorize(false));
    bias = bias - influence;
    debug();
    return true;
  }
};
