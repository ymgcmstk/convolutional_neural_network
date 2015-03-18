class InputLayer : public BaseLayer {
public:
  InputLayer () {
    layer_name = "InputLayer";
  }
  void initialize() {}
  bool forward () {
    return true;
  }
  bool backward () {
    return false;
  }
  void white () {
    //whitening用のdataが必要
  }
};
