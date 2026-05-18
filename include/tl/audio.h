#pragma once

#include <tl/tensor.h>

#include <string>
#include <vector>

namespace tl {
namespace audio {

struct WavData {
  std::vector<float> samples;
  int sample_rate;
};

// load a WAV file into a 1D tensor of float samples
WavData load_wav(const std::string& path);

// compute mel spectrogram from audio samples, convert to Tensor[1, frames, n_mels]
// samples is non-const because the underlying librosa-cpp call requires non-const ref
Tensor mel_spectrogram(std::vector<float>& samples,
                        int sr = 16000,
                        int n_fft = 400,
                        int n_hop = 160,
                        int n_mels = 80);
}
}
