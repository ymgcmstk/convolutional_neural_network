class IPLayer : public BaseLayer {
public:
  IPLayer () {
    layer_name = "IPLayer";
  }
  int num_output, w;
  MatrixXf weights, bias;
  float dropout;
  void initialize() {
    w = prev_layer->x * prev_layer->y * prev_layer->z;
    weights = MatrixXf::Random(num_output, w);
    bias = MatrixXf::Random(num_output, 1);
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
    prev_layer->influence.resize(prev_layer->data.rows(), prev_layer->data.cols());
    weights = weights - influence.asDiagonal() * (MatrixXf::Ones(num_output, 1) * prev_layer->vectorize(false));
    bias = bias - influence;
    debug();
    return true;
  }
};
