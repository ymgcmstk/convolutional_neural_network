class SoftmaxLayer : public BaseLayer {
public:
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    data = prev_layer->data.array().exp();
    data = data / data.sum();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    transposed = prev_layer->influence.transpose();
    diagonal = prev_layer->influence.asDiagonal();
    prev_layer->influence =
      (
        ((-prev_layer->influence/prev_layer->data.squaredNorm()) * transposed) + diagonal
      ) * influence;
    return true;
  }
private:
  MatrixXf transposed, diagonal;
};
