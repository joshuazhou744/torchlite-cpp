#include <tl/loss.h>
#include <tl/ops.h>
#include <tl/tensor.h>
#include <tl/factory.h>

#include <cstdint>

namespace tl {


// Mean square error loss: squares errors for a linear gradient wrt final prediction
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
  Tensor diff = sub(pred, target);
  Tensor sq = mul(diff, diff);
  std::vector<Tensor> steps;
  steps.push_back(sq);
  while (steps.back().sizes().size() > 0) {
    steps.push_back(sum(steps.back(), 0, false));
  }
  return scale(steps.back(), 1.0f / static_cast<float>(sq.numel()));
}

// Mean absolute error loss: constant gradient wrt final prediction, only knows direction of gradient
Tensor mae_loss(const Tensor& pred, const Tensor& target) {
  Tensor diff = sub(pred, target);
  Tensor abs_diff = abs(diff);
  std::vector<Tensor> steps;
  steps.push_back(abs_diff);
  while (steps.back().sizes().size() > 0) {
    steps.push_back(sum(steps.back(), 0, false));
  }
  return scale(steps.back(), 1.0f / static_cast<float>(abs_diff.numel()));
}

// L2 regularization: add a penalty that grows quadratically with weight size
Tensor l2_reg(const std::vector<Tensor*>& params, float lambda) {
  Tensor total = full({}, 0.0f); // scalar zero
  for (Tensor* p: params) {
    Tensor sq = mul(*p, *p);
    Tensor s = sq;
    while (s.sizes().size() > 0) s = sum(s, 0, false);
    total = add(total, s);
  }
  return scale(total, lambda);
}

// L1 regularization: add a penalty that grows linearly with weight size
Tensor l1_reg(const std::vector<Tensor*>& params, float lambda) {
  Tensor total = full({}, 0.0f);
  for (Tensor* p: params) {
    Tensor a = abs(*p);
    Tensor s = a;
    while (s.sizes().size() > 0) s = sum(s, 0, false);
    total = add(total, s);
  }
  return scale(total, lambda);
}

// Binary cross-entropy: penalty for confident wrong predictions in binary classification
// BCE = -mean(y*log(x) + (1-y)*log(1-x))
Tensor bce_loss(const Tensor& pred, const Tensor& target) {
  Tensor log_p = log(pred); // log(x)
  Tensor one = full(pred.sizes(), 1.0f);
  Tensor one_mp = sub(one, pred); // 1 - x
  Tensor one_mt = sub(one, target); // 1 - y
  Tensor log_1mp = log(one_mp); // log(1 - x)
  Tensor t1 = mul(target, log_p); // y * log(x)
  Tensor t2 = mul(one_mt, log_1mp); // (1 - y)(log(1-x))
  Tensor s = add(t1, t2);  // add both terms

  Tensor total = s;
  while (total.sizes().size() > 0) total = sum(total, 0, false);
  return scale(total, -1.0f / static_cast<float>(pred.numel())); // -mean(total)
}

// Negative log likelihood: correspond loss with a good guess (low loss) or bad guess (high loss)
// NLL = -mean(log_probs[i][y[i]]) over N samples
Tensor nll_loss(const Tensor& log_probs, const std::vector<int>& targets) {
  int64_t N = log_probs.sizes()[0]; // batch size
  int64_t C = log_probs.sizes()[1]; // number of classes

  // build mask for the correct class (listed in targets) in log_probs
  Tensor mask = full({N, C}, 0.0f);
  for (int64_t i = 0; i < N; ++i) {
    mask.data()[i * C + targets[i]] = 1.0f;
  }

  Tensor selected = mul(log_probs, mask); // element-wise multiplication to zero out non-target classes
  Tensor s = selected;
  while (s.sizes().size() > 0) s = sum(s, 0, false);
  return scale(s, -1.0f / static_cast<float>(N)); // -mean(selected)
}

// Cross-entropy loss: combines turns raw logits (unnormalized scores) into log probs for NLL
// loss = -mean(log(softmax(logits))[y]) over N samples
Tensor cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets) {
  Tensor probs = softmax(logits);
  Tensor log_probs = log(probs);
  return nll_loss(log_probs, targets);
}

} // namespace tl

