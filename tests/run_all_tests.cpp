#include <iostream>

void test_tensor();
void test_ops();

int main() {
  std::cout << "Running all tests... \n" << std::endl;

  test_tensor();
  test_ops();

  return 0;
}
