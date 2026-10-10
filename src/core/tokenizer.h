// SentencePiece detokenizer.  The vocabulary is embedded in the GGUF as an
// array of piece strings (tokenizer.ggml.tokens), so no tokenizer.model file is
// needed at runtime.
#pragma once

#include <string>
#include <vector>

namespace asr {

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(std::vector<std::string> pieces) : pieces_(std::move(pieces)) {}

    bool empty() const { return pieces_.empty(); }
    size_t size() const { return pieces_.size(); }

    // Detokenizes like SentencePieceProcessor::decode(): '▁' (U+2581) becomes a
    // space, <unk>/control pieces are dropped and leading/trailing spaces are
    // removed.
    std::string decode(const std::vector<int> & tokens) const;

    // Single piece as stored (for debugging).
    const std::string & piece(int id) const;

private:
    std::vector<std::string> pieces_;
};

}  // namespace asr
