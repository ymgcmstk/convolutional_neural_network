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
  void white () {
    //whitening用のdataが必要
  }
};

//入力画像(画像じゃなくてもいいけど)とラベルを保存する用
