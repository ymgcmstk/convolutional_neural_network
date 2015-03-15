class DataLayer : public BaseLayer {
public:
  void initialize() {}
  bool forward () {
    if (next_layer == NULL) return false;
    return true;
  }
  bool backward () {
    if (prev_layer == NULL) return false;
    return true;
  }
  bool calc_influence() {
    if (prev_layer == NULL) return false;
    //とりあえずcross entropy誤差
    prev_layer->influence = TODO;
    return true;
  }
  void white () {
    //whitening用のdataが必要
  }
};

//入力画像(画像じゃなくてもいいけど)とラベルを保存する用
