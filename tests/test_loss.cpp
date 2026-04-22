#include <iostream>
#include <cassert>
#include <cmath>
#include <tl/tensor.h>
#include <tl/loss.h>

static bool close(float a, float b, float e = 1e-4f) {
  return std::abs(a - b) < e;
}


void test_loss() {
  std::cout << "Running loss tests...\n";

  // mse: mean((pred - target)^2)
  {
    tl::Tensor pred({4});
    tl::Tensor target({4});
    pred.data()[0] = 2.0f; pred.data()[1] = 3.0f;
    pred.data()[2] = 4.0f; pred.data()[3] = 5.0f;
    target.data()[0] = 1.0f; target.data()[1] = 1.0f;
    target.data()[2] = 1.0f; target.data()[3] = 1.0f;
    // diffs: [1,2,3,4], sq: [1,4,9,16], mean = 30/4 = 7.5
    tl::Tensor loss = tl::mse_loss(pred, target);
    assert(close(loss.data()[0], 7.5f));
    std::cout << "  mse ok\n";
  }

  // mse backward: d/d(pred) = 2*(pred-target)/N
  {
    tl::Tensor pred({4});
    tl::Tensor target({4});
    for (int i = 0; i < 4; ++i) { pred.data()[i] = (float)(i+2);
target.data()[i] = 1.0f; }
    pred.set_requires_grad(true);

    tl::Tensor loss = tl::mse_loss(pred, target);
    loss.backward();

    // grad = 2*(pred-target)/N = 2*[1,2,3,4]/4 = [0.5,1.0,1.5,2.0]
    assert(close(pred.grad().data()[0], 0.5f));
    assert(close(pred.grad().data()[1], 1.0f));
    assert(close(pred.grad().data()[2], 1.5f));
    assert(close(pred.grad().data()[3], 2.0f));
    std::cout << "  mse backward ok\n";
  }

  // mae: mean(|pred - target|)
  {
    tl::Tensor pred({4});
    tl::Tensor target({4});
    pred.data()[0] = 3.0f; pred.data()[1] = -1.0f;
    pred.data()[2] = 2.0f; pred.data()[3] = 0.0f;
    target.data()[0] = 1.0f; target.data()[1] = 1.0f;
    target.data()[2] = 1.0f; target.data()[3] = 1.0f;
    // diffs: [2,-2,1,-1], abs: [2,2,1,1], mean = 6/4 = 1.5
    tl::Tensor loss = tl::mae_loss(pred, target);
    assert(close(loss.data()[0], 1.5f));
    std::cout << "  mae ok\n";
  }

  // mae backward: d/d(pred) = sign(pred-target)/N
  {
    tl::Tensor pred({4});
    tl::Tensor target({4});
    pred.data()[0] = 3.0f; pred.data()[1] = -1.0f;
    pred.data()[2] = 2.0f; pred.data()[3] = 0.5f;
    for (int i = 0; i < 4; ++i) target.data()[i] = 1.0f;
    pred.set_requires_grad(true);

    tl::Tensor loss = tl::mae_loss(pred, target);
    loss.backward();

    // diffs: [2,-2,1,-0.5], signs: [1,-1,1,-1], /4: [0.25,-0.25,0.25,-0.25]
    assert(close(pred.grad().data()[0],  0.25f));
    assert(close(pred.grad().data()[1], -0.25f));
    assert(close(pred.grad().data()[2],  0.25f));
    assert(close(pred.grad().data()[3], -0.25f));
    std::cout << "  mae backward ok\n";
  }

  std::cout << "loss tests passed.\n";
}

