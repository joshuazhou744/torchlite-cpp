#include <tl/audio.h>
#include <tl/ops.h>
#include <external/librosa.h>

#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace tl {
namespace audio {

// load a wav file into a tensor
WavData load_wav(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::invalid_argument("load_wav: cannot open file " + path);
  }

  // read WAV header for metadata
  // RIFF: Resource Interchange File Format, container for labeled chunk data in files
  // WAVE: content type identifier, confirms that the riff file has audio
  // fmt: mark the start of the wav labeled chunk ("fmt ", 4 chars)
  char riff[4], wave[4], fmt[4];
  // chunk_sizes: total files size minus 8 bytes (RIFF and chunk_size itself are the 8 bytes, unused anyways)
  // fmt_size: size of format chunk (typically 16 bytes)
  // data_size: size of raw audio data chunk
  uint32_t chunk_size, fmt_size, data_size;
  // audio_format: 1 = PCM (uncompressed audio), other values means compressed
  // num_channels: 1 = mono, 2 = stereo
  // bits_per_sample: bit depth per sample, how well defined each sample is
  uint16_t audio_format, num_channels, bits_per_sample;
  // sample_rate: samples per second, "resolution of audio"
  // byte_rate: bytes per second = sample_rate * num_channels * (bits_per_sample / 8)
  uint32_t sample_rate, byte_rate;
  // block_align: bytes per sample across all channels
  uint16_t block_align;

  // store first 4 bytes of file into riff[4]
  file.read(riff, 4);
  file.read(reinterpret_cast<char*>(&chunk_size), 4);

  // store next 4 bytes of file into wave[4]
  file.read(wave, 4);

  if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
    throw std::invalid_argument("load_wav: not a valid WAV file");
  }

  // store next 4 bytes of file into fmt[4]
  file.read(fmt, 4);
  // read 16 bytes of format data
  file.read(reinterpret_cast<char*>(&fmt_size), 4);
  file.read(reinterpret_cast<char*>(&audio_format), 2);
  file.read(reinterpret_cast<char*>(&num_channels), 2);
  file.read(reinterpret_cast<char*>(&sample_rate), 4);
  file.read(reinterpret_cast<char*>(&byte_rate), 4);
  file.read(reinterpret_cast<char*>(&block_align), 2);
  file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

  // skip extra fmt bytes if format chunk > standard 16 bytes
  if (fmt_size > 16) {
    // move file read position forward by a given amount of bytes starting from current position
    // jump past the extra format info
    file.seekg(fmt_size - 16, std::ios::cur);   }

  // find "data" chunk, skip non-data chunks (metadata, etc.)
  char chunk_id[4];
  while (file.read(chunk_id, 4)) {
    file.read(reinterpret_cast<char*>(&data_size), 4);
    if (std::string(chunk_id, 4) == "data") break;
    file.seekg(data_size, std::ios::cur);
  }

  // convert raw 16-bit samples to floats
  int64_t num_samples = data_size / (bits_per_sample / 8) / num_channels;
  std::vector<float> samples(num_samples);
  if (bits_per_sample == 16) {
    std::vector<int16_t> raw(num_samples * num_channels); // allocate space for raw 16-bit integers
    file.read(reinterpret_cast<char*>(raw.data()), data_size); // read all audio bytes
    for (int64_t i = 0; i < num_samples; ++i) {
      samples[i] = raw[i * num_channels] / 32768.0f; // normalize each sample
    }
  } else {
    throw std::invalid_argument("load_wav: only 16-bit WAV supported");
  }
  // return a WaveData struct
  return {samples, static_cast<int>(sample_rate)};
}

// take raw audio samples and return a tensor of audio sample
Tensor mel_spectrogram(const std::vector<float>& samples, int sr, int n_fft, int n_hop, int n_mels) {
  // compute mel spectrogram using librosacpp
  std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(
      samples,
      sr,
      n_fft,
      n_hop,
      "hann",
      true,
      "reflect",
      2.0f,
      n_mels,
      0,
      sr / 2
  );
  int64_t frames = mels.size(); // number of time steps
  int64_t mel_bins = mels[0].size(); // number of mel freq bins per frame

  // pack into tensor [1, frames, n_mels]
  Tensor out({1, frames, mel_bins});
  float* op = out.data();
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t m = 0; m < mel_bins; ++m) {
      op[f * mel_bins + m] = mels[f][m];
    }
  }
  return out;
}

}
}
