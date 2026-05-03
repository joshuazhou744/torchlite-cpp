#include <tl/tensor.h>
#include <tl/optim.h>
#include <tl/factory.h>

#include <iostream>
#include <cassert>
#include <cmath>

static bool close(float a, float b, float e = 1e-4f) {
  return std::abs(a - b) < e;
}

void test_optim() {
  std::cout << "Running optim tests...\n";

  // zero_grad: clears each param's gradient
  {
    tl::Tensor w({3});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f; w.data()[2] = 3.0f;
    w.set_requires_grad(true);
    w.grad() = tl::full({3}, 5.0f);  // pretend grad was populated

    tl::zero_grad({&w});
    assert(w.grad().empty());
    std::cout << "  zero_grad ok\n";
  }

  // SGD: w' = w - lr * grad
  {
    tl::Tensor w({2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.set_requires_grad(true);
    w.grad() = tl::Tensor({2});
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = 0.2f;

    tl::SGD opt({&w}, 0.5f);
    opt.step();
    // w = [1 - 0.5*0.1, 2 - 0.5*0.2] = [0.95, 1.9]
    assert(close(w.data()[0], 0.95f));
    assert(close(w.data()[1], 1.9f));
    std::cout << "  SGD basic ok\n";
  }

  // SGD with weight_decay: g' = g + wd*w; w' = w - lr * g'
  {
    tl::Tensor w({2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.set_requires_grad(true);
    w.grad() = tl::Tensor({2});
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = 0.2f;

    tl::SGD opt({&w}, 0.5f, 0.0f, 0.1f);
    opt.step();
    // g' = [0.1 + 0.1*1, 0.2 + 0.1*2] = [0.2, 0.4]
    // w  = [1 - 0.5*0.2, 2 - 0.5*0.4]  = [0.9, 1.8]
    assert(close(w.data()[0], 0.9f));
    assert(close(w.data()[1], 1.8f));
    std::cout << "  SGD weight_decay ok\n";
  }

  // SGD with momentum: v = momentum*v + g, w = w - lr*v
  {
    tl::Tensor w({2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.set_requires_grad(true);
    w.grad() = tl::Tensor({2});
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = 0.2f;

    tl::SGD opt({&w}, 0.5f, 0.9f);

    // step 1: v = g (first step), w = w - lr*g = [0.95, 1.9]
    opt.step();
    assert(close(w.data()[0], 0.95f));
    assert(close(w.data()[1], 1.9f));

    // step 2: same grad. v = 0.9*[0.1,0.2] + [0.1,0.2] = [0.19, 0.38]
    // w = [0.95 - 0.5*0.19, 1.9 - 0.5*0.38] = [0.855, 1.71]
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = 0.2f;
    opt.step();
    assert(close(w.data()[0], 0.855f));
    assert(close(w.data()[1], 1.71f));
    std::cout << "  SGD momentum ok\n";
  }

  // Adam: per-param adaptive lr
  // single step with default betas, t=1, bias correction divides by (1-beta^1)
  // m  = (1-beta1)*g,  m_hat = m / (1-beta1) = g
  // v  = (1-beta2)*g², v_hat = v / (1-beta2) = g²
  // w -= lr * g / (sqrt(g²) + eps) ≈ lr * sign(g)
  {
    tl::Tensor w({2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.set_requires_grad(true);
    w.grad() = tl::Tensor({2});
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = -0.2f;

    tl::Adam opt({&w}, 0.01f);
    opt.step();
    // w[0] ≈ 1 - 0.01 * (0.1 / 0.1) = 0.99
    // w[1] ≈ 2 - 0.01 * (-0.2 / 0.2) = 2.01
    assert(close(w.data()[0], 0.99f, 1e-3f));
    assert(close(w.data()[1], 2.01f, 1e-3f));
    std::cout << "  Adam ok\n";
  }

  // AdamW: decoupled weight decay
  // w -= lr * (adam_update + weight_decay * w)
  // single step: adam_update ≈ sign(g), so w -= lr*(sign(g) + wd*w)
  {
    tl::Tensor w({2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.set_requires_grad(true);
    w.grad() = tl::Tensor({2});
    w.grad().data()[0] = 0.1f; w.grad().data()[1] = -0.2f;

    tl::AdamW opt({&w}, 0.01f, 0.9f, 0.999f, 1e-8f, 0.1f);
    opt.step();
    // w[0] ≈ 1 - 0.01 * (1.0 + 0.1*1) = 1 - 0.011 = 0.989
    // w[1] ≈ 2 - 0.01 * (-1.0 + 0.1*2) = 2 - 0.01*(-0.8) = 2.008
    assert(close(w.data()[0], 0.989f, 1e-3f));
    assert(close(w.data()[1], 2.008f, 1e-3f));
    std::cout << "  AdamW ok\n";
  }

  // multi-step convergence sanity check: SGD on f(w) = (w - 5)^2
  // grad = 2*(w - 5); converges to w = 5
  {
    tl::Tensor w({1});
    w.data()[0] = 0.0f;
    w.set_requires_grad(true);

    tl::SGD opt({&w}, 0.1f);
    for (int i = 0; i < 100; ++i) {
      w.grad() = tl::Tensor({1});
      w.grad().data()[0] = 2.0f * (w.data()[0] - 5.0f);
      opt.step();
    }
    assert(close(w.data()[0], 5.0f, 1e-2f));
    std::cout << "  SGD convergence ok\n";
  }

  std::cout << "Optim tests passed.\n\n";
}
