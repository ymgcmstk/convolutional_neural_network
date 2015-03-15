class IPLayer : public BaseLayer {
public:
  int wx, wy;
  MatrixXf weights;
  float dropout;
  void initialize() {
    wy = prev_layer->x * prev_layer->y * prev_layer->z;
    weights = MatrixXf::Random(wx, wy);
  }
  bool forward () {
    if (next_layer == NULL) return false;
    data = weights * prev_layer->vectorize();
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    //TODO
    return true;
  }
};
