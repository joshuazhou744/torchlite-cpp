#include <iostream>

void test_tensor();
void test_ops();
void test_activation();
void test_factory();
void test_nn();
void test_autograd();
void test_loss();
void test_optim();

int main() {
  std::cout << "Running all tests... \n" << std::endl;

  test_tensor();
  test_ops();
  test_activation();
  test_factory();
  test_nn();
  test_autograd();
  test_loss();
  test_optim();

  return 0;
}
