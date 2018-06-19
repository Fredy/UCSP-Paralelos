#include "blur.cuh"
#include "grayScale.cuh"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << argv[0] << ": needs two arguments\n"
         << "<image_path> <option>\n";
    return 0;
  }

  string image_path(argv[1]), option(argv[2]);
  if (option == "gray") {
    grayScale(image_path);
  } else {
    imageBlur(image_path);
  }
  return 0;
}
