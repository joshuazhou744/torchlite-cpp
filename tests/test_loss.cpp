#include <tl/tensor.h>
#include <tl/loss.h>

#include <iostream>
#include <cassert>
#include <cmath>

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

  // bce: -mean(y*log(x) + (1-y)*log(1-x))
  {
    tl::Tensor pred({2});
    tl::Tensor target({2});
    pred.data()[0] = 0.9f; pred.data()[1] = 0.1f;
    target.data()[0] = 1.0f; target.data()[1] = 0.0f;
    // -mean(1*log(0.9) + 0 + 0 + 1*log(0.9)) = -log(0.9) ≈ 0.10536
    tl::Tensor loss = tl::bce_loss(pred, target);
    assert(close(loss.data()[0], -std::log(0.9f), 1e-3f));
    std::cout << "  bce ok\n";
  }

  // nll: -mean(log_probs[i][y[i]])
  {
    tl::Tensor log_probs({2, 3});
    // sample 0: log_probs = [-1.0, -2.0, -3.0]; correct=0  ->  -(-1.0) = 1.0
    // sample 1: log_probs = [-0.5, -1.5, -2.5]; correct=2  ->  -(-2.5) = 2.5
    // mean = (1.0 + 2.5)/2 = 1.75
    log_probs.data()[0] = -1.0f; log_probs.data()[1] = -2.0f; log_probs.data()[2] =
  -3.0f;
    log_probs.data()[3] = -0.5f; log_probs.data()[4] = -1.5f; log_probs.data()[5] =
  -2.5f;
    std::vector<int> targets = {0, 2};
    tl::Tensor loss = tl::nll_loss(log_probs, targets);
    assert(close(loss.data()[0], 1.75f));
    std::cout << "  nll ok\n";
  }

  // cross_entropy: softmax + log + nll
  // uniform logits -> softmax = uniform 1/C -> log(1/3) for each class
  // loss = -log(1/3) ≈ 1.0986
  {
    tl::Tensor logits({2, 3});
    for (int i = 0; i < 6; ++i) logits.data()[i] = 1.0f;
    std::vector<int> targets = {0, 1};
    tl::Tensor loss = tl::cross_entropy_loss(logits, targets);
    assert(close(loss.data()[0], -std::log(1.0f / 3.0f), 1e-3f));
    std::cout << "  cross_entropy ok\n";
  }

  // l2_reg: lambda * sum(w^2)
  {
    tl::Tensor w({3});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f; w.data()[2] = 3.0f;
    // sum(w^2) = 1+4+9 = 14, * 0.5 = 7.0
    tl::Tensor reg = tl::l2_reg({&w}, 0.5f);
    assert(close(reg.data()[0], 7.0f));
    std::cout << "  l2_reg ok\n";
  }

  // l1_reg: lambda * sum(|w|)
  {
    tl::Tensor w({3});
    w.data()[0] = -1.0f; w.data()[1] = 2.0f; w.data()[2] = -3.0f;
    // sum(|w|) = 1+2+3 = 6, * 0.5 = 3.0
    tl::Tensor reg = tl::l1_reg({&w}, 0.5f);
    assert(close(reg.data()[0], 3.0f));
    std::cout << "  l1_reg ok\n";
  }


  std::cout << "loss tests passed.\n";
}

