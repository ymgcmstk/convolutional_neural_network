#include <iostream>
#include <fstream>
#include <string>
#include "../Layer/input_layer.cpp"
#include "../Layer/loss/softmax_loss_layer.cpp"
#include "../Layer/loss/cross_entropy_loss_layer.cpp"
#include "../Layer/loss/euclidean_loss_layer.cpp"
#include "../Layer/prelu_layer.cpp"
#include "../Layer/relu_layer.cpp"
#include "../Layer/softmax_layer.cpp"
#include "../Layer/ip_layer.cpp"

void set_data_size (BaseLayer* layer, int input_x, int input_y, int input_z) {
  layer->x = input_x;
  layer->y = input_y;
  layer->z = input_z;
}

//init_model_and_data<float>(fname, model, data);
void init_model_and_data (const string& fname, BaseStructure& bs) {
    ifstream file (fname.c_str(), ios::binary);
    string buf_string, layer_name;
    BaseLayer *new_layer, *prev_layer;
    int input_x, input_y, input_z, wx;
    float learning_rate;
    if (! file) {
      cout << "ErrorErrorErrorError(init_model_and_data)" << endl;
      cout << fname << " does not exist." << endl;
      exit(1);
    }
    while (getline(file, buf_string)) {
      layer_name.assign(buf_string, 0, buf_string.find(" "));
      if (layer_name == "Input") {
        sscanf(buf_string.c_str(), "Input %d %d %d", &input_x, &input_y, &input_z);
        new_layer = new InputLayer();
        bs.input_layer = new_layer;
        set_data_size(new_layer, input_x, input_y, input_z);
        prev_layer = new_layer;
      } else if (layer_name == "SoftmaxLoss") {
        sscanf(buf_string.c_str(), "SoftmaxLoss %f", &learning_rate);
        new_layer = new SoftmaxLossLayer();
        new_layer->learning_rate = learning_rate;
        bs.output_layer = new_layer;
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "EuclideanLoss") {
        sscanf(buf_string.c_str(), "EuclideanLoss %f", &learning_rate);
        new_layer = new EuclideanLossLayer();
        new_layer->learning_rate = learning_rate;
        bs.output_layer = new_layer;
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "CrossEntropyLoss") {
        sscanf(buf_string.c_str(), "CrossEntropyLoss %f", &learning_rate);
        new_layer = new CrossEntropyLossLayer();
        new_layer->learning_rate = learning_rate;
        bs.output_layer = new_layer;
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "IPLayer") {
        sscanf(buf_string.c_str(), "IPLayer %d", &wx);
        IPLayer *new_ip;
        new_ip = new IPLayer();
        new_ip->wx = wx;
        new_layer = new_ip;
        input_x = wx;
        input_y = 1;
        input_z = 1;
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
        prev_layer = new_layer;
        new_layer->initialize();
      } else if (layer_name == "ReLULayer") {
        new_layer = new ReLULayer();
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
        prev_layer = new_layer;
      } else if (layer_name == "PReLULayer") {
        new_layer = new PReLULayer();
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
        prev_layer = new_layer;
      } else if (layer_name == "SoftmaxLayer") {
        new_layer = new SoftmaxLayer();
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
        prev_layer = new_layer;
      } else {
        cout << "ErrorErrorErrorError(init_model_and_data)" << endl;
        cout << layer_name  << " does not exist." << endl;
        exit(1);
      }
    }
    bs.last_layer = bs.output_layer->prev_layer;
}
