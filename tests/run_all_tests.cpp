#include <iostream>

void test_tensor();
void test_ops();
void test_activation();
void test_factory();

int main() {
  std::cout << "Running all tests... \n" << std::endl;

  test_tensor();
  test_ops();
  test_activation();
  test_factory();

  return 0;
}
