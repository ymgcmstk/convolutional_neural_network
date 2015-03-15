#include "../Layer/base_layer.hpp"

class BaseStructure {
public:
  ~BaseStructure () {}
  void forward (){
    BaseLayer* updating_layer = input_layer;
    while (updating_layer->forward()) {
      updating_layer = updating_layer->next_layer;
    }
  }
  void backward (){
    //first(入力層の次もしくは出力層の手前の層のパラメータ)から順に計算していく
    BaseLayer* updating_layer = output_layer;
    while (updating_layer->backward()) {
      updating_layer = updating_layer->prev_layer;
    }
  }
  void calc_influence () {
    BaseLayer* updating_layer = output_layer;
    while (updating_layer->calc_influence()) {
      updating_layer = updating_layer->prev_layer;
    }
  }

  void print_output () {
    last_layer->print();
  }
  BaseLayer *input_layer;
  BaseLayer *output_layer, *last_layer;
};
