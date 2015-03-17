class ConvLayer : public BaseLayer {
public:
  ConvLayer () {
    layer_name = "ConvLayer";
  }
  int wx, wy;
  MatrixXf weights, bias;
  float dropout;
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

/*
方針:
data中の対象領域、つまり
  (offset_x * kernel_size, offset_y)と(offset_x * kernel_size + kernel_size * input_z, offset_y + kernel_size)によって囲まれる領域
をvectorizeして、それに左から重みをかけてbiasをたすという方針で
端っこどうすればいいのかはよくわかってない
http://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cppを参照


weightを3*3とする、これは奥行きがinput_zの9個のvectorの集合と見なせる
答えの行列を0で初期化
9個のうちのあるベクトルについて
  各pixelにそれをそれぞれかけていき出力をmap化する
  offset分ずらして答えの行列に足す
答えの行列に適切にbiasをたす

 */
