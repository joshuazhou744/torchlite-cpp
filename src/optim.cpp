#include <tl/optim.h>
#include <tl/tensor.h>
#include <tl/ops.h>

#include <cstdint>
#include <cmath>

namespace tl {

// Helper functions
void zero_grad(const std::vector<Tensor*>& params) {
  for (Tensor* p: params) {
    if (p->requires_grad) p->grad() = Tensor();
  }
}

// SGD: Stochastic gradient descent with L2 regularization (weight decay)
// weight_decay shrinks weights towards zero each step, equivalent to adding the lambda*sum(w^2) penalty
SGD::SGD(
    const std::vector<Tensor*>& params,
    float lr,
    float momentum,
    float weight_decay
  ) : params_(params),
      lr_(lr),
      momentum_(momentum),
      weight_decay_(weight_decay)
{}

// SGD step
void SGD::step() {
  for (Tensor* p: params_) {
    if (!p->requires_grad || p->grad().empty()) continue;

    Tensor g = p->grad();
    // weight decay: g' = g + weight_decay * w
    if (weight_decay_ != 0.0f) {
      g = add(g, scale(*p, weight_decay_));
    }

    // momentum: v' = momentum * v + g
    if (momentum_ != 0.0f) {
      auto it = velocity_.find(p);
      if (it == velocity_.end()) {
        velocity_[p] = g; // first step, v = g
      } else {
        velocity_[p] = add(scale(it->second, momentum_), g);
      }
      g = velocity_[p];
    }

    // w' = w - lr * g
    *p = sub(*p, scale(g, lr_));
  }
}

// SGD zero grads
void SGD::zero_grad() {
  tl::zero_grad(params_); // explicitly use tl namespace to prevent infinite recursion
}

// Adam: use first moment and second moment to give each param an adaptive learning rate
// params with steady gradients take larger steps
// parms with noisy gradients take smaller steps
// for each time step t:
// m' = beta1 * m + (1 - beta1) * g : first moment (gradient avg)
// v' = beta2 * v + (1 - beta2) * g^2 : second moment (squared grad avg)
// m_hat = m / (1 - beta1^t) : bias correction for step direction
// v_hat = v / (1 - beta2^t) : bias correction for step magnitude
// w' = w - lr * m_hat / (sqrt(v_hat) + eps)
Adam::Adam(
    const std::vector<Tensor*>& params,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) : params_(params),
    lr_(lr),
    beta1_(beta1),
    beta2_(beta2),
    eps_(eps),
    weight_decay_(weight_decay)
{}

// Adam step
void Adam::step() {
  ++t_;
  for (Tensor* p: params_) {
    if (!p->requires_grad || p->grad().empty()) continue;

    Tensor g = p->grad();

    // L2-like weight decay
    if (weight_decay_ != 0.0f) {
      g = add(g, scale(*p, weight_decay_));
    }

    // first/second moment updates
    auto m_it = m_.find(p);
    if (m_it == m_.end()) {
      m_[p] = scale(g, 1.0f - beta1_);
    } else {
      m_[p] = add(scale(m_it->second, beta1_), scale(g, 1.0f - beta1_));
    }

    Tensor g_sq = mul(g, g);
    auto v_it = v_.find(p);
    if (v_it == v_.end()) {
      v_[p] = scale(g_sq, 1.0f - beta2_);
    } else {
      v_[p] = add(scale(v_it->second, beta2_), scale(g_sq, 1.0f - beta2_));
    }

    // bias correction: early steps underestimate moments (starting at 0), so we rescale
    float bc1 = 1.0f -std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f -std::pow(beta2_, static_cast<float>(t_));
    Tensor m_hat = scale(m_[p], 1.0f / bc1);
    Tensor v_hat = scale(v_[p], 1.0f / bc2);

    // w' = w - lr * m_hat / (sqrt(v_hat) + eps)
    // dividing by sqrt(v_hat) shrinks the step for params with large/noisy gradients
    Tensor denom = add(sqrt(v_hat), full(v_hat.sizes(), eps_));
    Tensor update = div(m_hat, denom);
    *p = sub(*p, scale(update, lr_));
  }
}

// Adam zero grad
void Adam::zero_grad() {
  tl::zero_grad(params_);
}

// AdamW: same as Adam except weight decay is decoupled and applied to weights directly rather than the gradient
// g' = g (no weight decay)
// w' = w - lr * update - lr * weight_decay * w
AdamW::AdamW(
    const std::vector<Tensor*>& params,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) : params_(params),
    lr_(lr),
    beta1_(beta1),
    beta2_(beta2),
    eps_(eps),
    weight_decay_(weight_decay)
{}

// AdamW step
void AdamW::step() {
  ++t_;
  for (Tensor* p: params_) {
    if (!p->requires_grad || p->grad().empty()) continue;

    Tensor g = p->grad(); // raw gradient, no weight decay

    // first/second moment updates (same as Adam)
    auto m_it = m_.find(p);
    if (m_it == m_.end()) {
      m_[p] = scale(g, 1.0f - beta1_);
    } else {
      m_[p] = add(scale(m_it->second, beta1_), scale(g, 1.0f - beta1_));
    }

    Tensor g_sq = mul(g, g);
    auto v_it = v_.find(p);
    if (v_it == v_.end()) {
      v_[p] = scale(g_sq, 1.0f - beta2_);
    } else {
      v_[p] = add(scale(v_it->second, beta2_), scale(g_sq, 1.0f - beta2_));
    }

    // bias correction: early steps underestimate moments (starting at 0), so we rescale
    // (same as Adam)
    float bc1 = 1.0f -std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f -std::pow(beta2_, static_cast<float>(t_));
    Tensor m_hat = scale(m_[p], 1.0f / bc1);
    Tensor v_hat = scale(v_[p], 1.0f / bc2);

    // w' = w - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * w
    // dividing by sqrt(v_hat) shrinks the step for params with large/noisy gradients
    Tensor denom = add(sqrt(v_hat), full(v_hat.sizes(), eps_));
    Tensor adam_update = div(m_hat, denom);
    *p = sub(*p, scale(add(adam_update, scale(*p, weight_decay_)), lr_));
  }
}

// Adam zero grad
void AdamW::zero_grad() {
  tl::zero_grad(params_);
}

} // namespace tl
