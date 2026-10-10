// End-to-end recognizer: log-mel -> Conformer encoder -> RNN-T greedy decoder
// -> SentencePiece text.
//
// Two modes:
//  * transcribe(): offline, the whole buffer in one encoder pass;
//  * begin()/feed()/finish(): push-to-talk streaming.  The encoder is not
//    causal (it was trained with full bidirectional attention), so streaming is
//    done by re-encoding a sliding window of the last `max_window_sec` seconds
//    and committing the stable token prefix with a LocalAgreement-2 rule
//    (the same idea whisper-stream uses).  Partial text appears ~0.7-1 s after
//    the first words, and the final pass on release is exact.
#pragma once

#include <string>
#include <vector>

#include "core/encoder.h"
#include "core/mel.h"
#include "core/model.h"
#include "core/rnnt.h"
#include "core/tokenizer.h"

namespace asr {

// Worker threads used when a configuration leaves n_threads at 0 ("auto"):
// half of the logical processors, at least one.  ggml's thread pool busy-waits
// in its barriers, so one worker per logical processor oversubscribes hybrid
// P/E-core CPUs and the encoder can stall for minutes under a power limit;
// half of the logical processors still keeps every core busy.
int default_n_threads();

struct RecognizerConfig {
    std::string model_path;
    int n_threads = 0;              // 0 = default_n_threads()

    // streaming
    float min_window_sec = 0.5f;    // do not run the encoder before this
    float max_window_sec = 20.0f;   // window length for long dictations
    float step_sec = 1.0f;          // re-encode every N seconds of new audio
    float commit_guard_sec = 0.4f;  // never commit text from the last N seconds
};

struct Transcription {
    std::string text;
    std::vector<int> tokens;
    double audio_sec = 0.0;
    double infer_ms = 0.0;
    int n_windows = 0;
};

class Recognizer {
public:
    Recognizer();
    ~Recognizer();

    bool init(const RecognizerConfig & cfg, std::string * err);

    const HParams & hp() const { return model_.hp(); }
    const Tokenizer & tokenizer() const { return tokenizer_; }
    const RnntDecoder & rnnt() const { return rnnt_; }
    const std::string & model_path() const { return cfg_.model_path; }
    bool loaded() const { return loaded_; }

    void set_n_threads(int n);
    int n_threads() const { return cfg_.n_threads; }

    // ---------------- offline
    bool transcribe(const float * pcm, int n_samples, Transcription * out, std::string * err);

    // ---------------- streaming (push-to-talk)
    void begin();
    // Appends audio; may run the encoder.  Cheap when nothing is due.
    void feed(const float * pcm, int n_samples);
    // Current text: committed (stable) and provisional (may still change).
    void snapshot(std::string * committed, std::string * provisional) const;
    // Final pass over everything recorded; returns the exact transcription.
    bool finish(Transcription * out, std::string * err);
    bool recording() const { return recording_; }
    double audio_seconds() const {
        return static_cast<double>(buffer_.size()) / hp().sample_rate;
    }
    // Full text of the current hypothesis (committed + provisional).
    std::string current_text() const;

private:
    struct TokFrame {
        int frame = 0;  // absolute mel-frame index
        int token = 0;
    };

    void update_mel();
    bool run_window(int f0, int f1, std::vector<int> * tokens, std::vector<int> * frames, double * ms,
                    std::string * err);
    void streaming_step(bool force);
    std::string decode_tokens(const std::vector<TokFrame> & t) const;

    RecognizerConfig cfg_;
    Model model_;
    MelFrontend mel_;
    Encoder encoder_;
    RnntDecoder rnnt_;
    Tokenizer tokenizer_;
    bool loaded_ = false;

    // audio + mel cache (whole recording)
    std::vector<float> buffer_;
    std::vector<float> mel_cache_;   // [mel_frames * n_mels], time-major
    int mel_frames_ = 0;
    int mel_covered_samples_ = 0;

    // streaming state
    bool recording_ = false;
    int last_run_samples_ = 0;
    int window_f0_ = 0;              // first mel frame of the current window
    std::vector<TokFrame> committed_;
    std::vector<TokFrame> hypothesis_;   // previous window hypothesis
    std::vector<int> prev_tokens_;
    int n_windows_ = 0;
    double last_infer_ms_ = 0.0;
};

}  // namespace asr
