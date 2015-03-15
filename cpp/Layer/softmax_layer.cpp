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
    return true;
  }
  bool calc_influence() {
    return true;
  }
};
