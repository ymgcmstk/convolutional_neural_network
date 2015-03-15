#include <iostream>
#include <fstream>

using namespace std;

#include <arpa/inet.h>

#define ERROR() do {                                            \
    cout << "I/O error at " << __func__ << ": " << __LINE__ <<  \
      " (offset " << file.tellg() << ")" << endl;               \
    return;                                                     \
  } while (0)

static void read_mnist(const string &full_path, const string &saving_path)
{
  ifstream file (full_path.c_str(), ios::binary);
  if ( ! file)
    ERROR();

  int magic_number=0;
  int number_of_images=0;
  int n_rows=0;
  int n_cols=0;
  if ( ! file.read((char*)&magic_number,sizeof(magic_number)))
    ERROR();
  magic_number= ntohl(magic_number);
  if ( ! file.read((char*)&number_of_images,sizeof(number_of_images)))
    ERROR();
  number_of_images= ntohl(number_of_images);
  if ( ! file.read((char*)&n_rows,sizeof(n_rows)))
    ERROR();
  n_rows= ntohl(n_rows);
  if ( ! file.read((char*)&n_cols,sizeof(n_cols)))
    ERROR();
  n_cols= ntohl(n_cols);

  ofstream ofs;
  ofs.open(saving_path.c_str(), ios::out|ios::binary|ios::trunc);
  for(long i=0;i<number_of_images*n_rows*n_cols;++i) {
    unsigned char temp=0;
    if ( ! file.read((char*)&temp,sizeof(temp))) ERROR();
    ofs.write((char *) &temp, sizeof(unsigned char));
  }
  ofs.close();
}

int main(int argc, char *argv[]) {
  cout << "Process [" << argv[1] << "]" << endl;
  read_mnist(argv[1], argv[2]);
}
