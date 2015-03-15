#include <Eigen/Dense>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <time.h>

void print_image_and_label(unsigned long n);

string train_images = "/Users/yamaguchi/myproject/my_cnn/mnist_data/train-images-binary.txt";
string train_labels = "/Users/yamaguchi/myproject/my_cnn/mnist_data/train-labels-binary.txt";

MatrixXf access_image (unsigned long n) {
  MatrixXf mat(28, 28);
  FILE *fp;
  unsigned char buf[784];
  fp = fopen(train_images.c_str(), "rb");
  if (! fp) {
    cout << "file not found" << endl;
    exit(1);
  }
  //cout << n << endl;
  //print_image_and_label(n);
  fseek(fp, sizeof(unsigned char) * n * 28 * 28, SEEK_SET);
  fread(buf, sizeof(unsigned char), 28 * 28, fp);
  fclose(fp);
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      mat(i, j) = float(buf[i * 28 + j]);
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
  srand((unsigned)time(NULL));
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