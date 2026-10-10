#include "core/tokenizer.h"

#include <cstdio>
#include <cstdlib>

namespace asr {

namespace {
const std::string kEmpty;

// U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's word marker
const char kWordMarker[] = "\xe2\x96\x81";

bool is_control_piece(const std::string & p) {
    // <unk>, <s>, </s>, ... but not single characters
    return p.size() >= 3 && p.front() == '<' && p.back() == '>';
}
}  // namespace

const std::string & Tokenizer::piece(int id) const {
    if (id < 0 || id >= static_cast<int>(pieces_.size())) return kEmpty;
    return pieces_[static_cast<size_t>(id)];
}

std::string Tokenizer::decode(const std::vector<int> & tokens) const {
    std::string out;
    out.reserve(tokens.size() * 2);
    for (int id : tokens) {
        const std::string & p = piece(id);
        if (p.empty() || is_control_piece(p)) continue;
        size_t i = 0;
        while (i < p.size()) {
            if (p.compare(i, 3, kWordMarker) == 0) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(p[i]);
                ++i;
            }
        }
    }
    // SentencePiece strips a single leading space produced by the first marker
    size_t b = 0, e = out.size();
    while (b < e && out[b] == ' ') ++b;
    while (e > b && out[e - 1] == ' ') --e;
    return out.substr(b, e - b);
}

}  // namespace asr
