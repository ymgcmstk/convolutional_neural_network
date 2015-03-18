#include <Eigen/Dense>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <time.h>

void print_image_and_label(unsigned long n);
unsigned long get_random_long ();

string train_images = "/Users/yamaguchi/myproject/my_cnn/mnist_data/train-images-binary.txt";
string train_labels = "/Users/yamaguchi/myproject/my_cnn/mnist_data/train-labels-binary.txt";

MatrixXf access_image (unsigned long n) {
  MatrixXf mat(784, 1);
  FILE *fp;
  unsigned char buf[784];
  fp = fopen(train_images.c_str(), "rb");
  if (! fp) {
    cout << "file not found" << endl;
    exit(1);
  }
  fseek(fp, sizeof(unsigned char) * n * 28 * 28, SEEK_SET);
  fread(buf, sizeof(unsigned char), 28 * 28, fp);
  fclose(fp);
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      mat(i * 28 + j, 0) = float(buf[i * 28 + j]);
    }
  }
  return mat;
}

MatrixXf access_label (unsigned long n) {
  MatrixXf mat = MatrixXf::Zero(10, 1);
  FILE *fp;
  unsigned char buf;
  fp = fopen(train_labels.c_str(), "rb");
  if (! fp) {
    cout << "file not found" << endl;
    exit(1);
  }
  fseek(fp, sizeof(unsigned char) * n, SEEK_SET);
  fread(&buf, sizeof(unsigned char), 1, fp);
  fclose(fp);
  mat(int(buf), 0) = 1;
  return mat;
}

MatrixXf calc_mean (int n) {
  MatrixXf mean = MatrixXf::Zero(28 * 28, 1);
  for (int i = 0; i < n ; i++) {
    mean += access_image(i);
  }
  return mean / n;
}

MatrixXf calc_var (int n, MatrixXf mean) {
  return MatrixXf::Zero(28 * 28, 1);
}

float calc_mean_float (int n) {
  float mean;
  for (int i = 0; i < n ; i++) {
    mean += access_image(i).mean();
  }
  return mean / n;
}

float calc_var_float (int n, MatrixXf mean) {
  float var = 0;
  MatrixXf temp;
  for (int i = 0; i < n ; i++) {
    var += (access_image(i)-mean).squaredNorm() / (28 * 28);
  }
  return var / n;
}

//for debugging
void print_image_and_label (unsigned long n) {
  FILE *fp;
  unsigned char buf_label;
  unsigned char buf_image[784];

  fp = fopen(train_labels.c_str(), "rb");
  if (! fp) {
    cout << "file not found" << endl;
    exit(1);
  }
  fseek(fp, sizeof(unsigned char) * n, SEEK_SET);
  fread(&buf_label, sizeof(unsigned char), 1, fp);
  fclose(fp);
  cout << float(buf_label) << endl;

  cout << "image : " << endl;
  fp = fopen(train_images.c_str(), "rb");
  if (! fp) {
    cout << "file not found" << endl;
    exit(1);
  }
  fseek(fp, sizeof(unsigned char) * n * 28 * 28, SEEK_SET);
  fread(buf_image, sizeof(unsigned char), 28 * 28, fp);
  fclose(fp);
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      if (float(buf_image[i * 28 + j]) > 0) cout << "x ";
      else cout << "  ";
    }
    cout << endl;
  }
}

//for debugging
unsigned long get_random_long () {
  //srand((unsigned)time(NULL));
  unsigned long temp;
  while (true) {
    temp = (rand() % 2) * RAND_MAX + rand();
    if (temp < 60000) {
      cout << "generated long:" << endl;
      cout << temp << endl;
      return temp;
    }
  }
}
