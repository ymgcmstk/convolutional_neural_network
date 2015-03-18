#include <iostream>
#include <fstream>
#include <string>

// a layer for data
#include "../Layer/input_layer.cpp"
// layers for loss
#include "../Layer/loss/softmax_loss_layer.cpp"
#include "../Layer/loss/cross_entropy_loss_layer.cpp"
#include "../Layer/loss/euclidean_loss_layer.cpp"
// layers for activation
#include "../Layer/prelu_layer.cpp"
#include "../Layer/relu_layer.cpp"
#include "../Layer/softmax_layer.cpp"
// a layer for pooling
#include "../Layer/pool/max_pooling_layer.cpp"
// other layers
#include "../Layer/ip_layer.cpp"
#include "../Layer/conv_layer.cpp"

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
    int input_x, input_y, input_z, num_output, kernel_x, kernel_y, stride, pad_x, pad_y;
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
        //この辺input_x, input_y, input_zをすべて1にしたほうが正しいだろうか、意味ないけど
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "EuclideanLoss") {
        sscanf(buf_string.c_str(), "EuclideanLoss %f", &learning_rate);
        new_layer = new EuclideanLossLayer();
        new_layer->learning_rate = learning_rate;
        bs.output_layer = new_layer;
        //この辺input_x, input_y, input_zをすべて1にしたほうが正しいだろうか、意味ないけど
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "CrossEntropyLoss") {
        sscanf(buf_string.c_str(), "CrossEntropyLoss %f", &learning_rate);
        new_layer = new CrossEntropyLossLayer();
        new_layer->learning_rate = learning_rate;
        bs.output_layer = new_layer;
        //この辺input_x, input_y, input_zをすべて1にしたほうが正しいだろうか、意味ないけど
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
      } else if (layer_name == "ConvLayer") {
        sscanf(buf_string.c_str(),
               "ConvLayer %d %d %d",
               &kernel_x,
               &kernel_y,
               &num_output
        );
        stride = 1;
        pad_x = stride / 2;
        pad_y = stride / 2;
        ConvLayer *new_conv;
        new_conv = new ConvLayer();
        new_conv->stride = stride;
        new_conv->pad_x = pad_x;
        new_conv->pad_y = pad_y;
        new_conv->kernel_x = kernel_x;
        new_conv->kernel_y = kernel_y;
        new_layer = new_conv;
        //strideは1, padはkernel_size / 2で固定する、ややこしくなるからそれでいきたいところ
        input_x = (input_x + stride - 1) / stride;
        input_y = (input_y + stride - 1) / stride;
        input_z = num_output;
        set_data_size(new_layer, input_x, input_y, input_z);
        new_layer->prev_layer = prev_layer;
        new_layer->prev_layer->next_layer = new_layer;
        prev_layer = new_layer;
        new_layer->initialize();
      } else if (layer_name == "IPLayer") {
        sscanf(buf_string.c_str(), "IPLayer %d", &num_output);
        IPLayer *new_ip;
        new_ip = new IPLayer();
        new_ip->num_output = num_output;
        new_layer = new_ip;
        input_x = num_output;
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
