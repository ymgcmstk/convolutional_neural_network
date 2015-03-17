class SoftmaxLayer : public BaseLayer {
public:
  SoftmaxLayer () {
    layer_name = "SoftmaxLayer";
  }
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    data = prev_layer->data.array().exp();
    data = data / data.sum();
    debug();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    transposed = data.transpose();
    diagonal = data.asDiagonal();
    prev_layer->influence =
      (
        diagonal - ((prev_layer->data/prev_layer->data.squaredNorm()) * transposed)
      ) * influence;
    debug();
    return true;
  }
private:
  MatrixXf transposed, diagonal;
};
