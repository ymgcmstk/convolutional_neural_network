class PReLULayer : public BaseLayer {
public:
  float weight;
  void initialize() {
    weight = 0;
  }
  bool forward() {
    if (next_layer == NULL) return false;
    data = prev_layer->data * weight;
    data = prev_layer->data.array().max(data.array());
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    transposed = influence.transpose();
    compared = (prev_layer->vectorize().array()<=0.0).cast<float>().array() *
      prev_layer->vectorize().array();
    prev_layer->influence =
      (prev_layer->data.array()>0.0).cast<float>().array() * influence.array() +
      (prev_layer->data.array()<=0.0).cast<float>().array() * influence.array() * weight;
    weight = weight - (transposed * compared)(0,0);
    return true;
  }
private:
  MatrixXf transposed;
  MatrixXf compared;
};
