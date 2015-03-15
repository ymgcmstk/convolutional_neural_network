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
    return true;
  }
};
