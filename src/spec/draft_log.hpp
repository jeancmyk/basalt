// src/spec/draft_log.hpp - the reader for --draft-log's window lines.
//
// `--draft-log FILE` writes one `W` line per decode window (generate.cpp), and until this reader existed nothing read
// them back.  The format is positional and append-only:
//
//     W <T> <accepted> <from_sfx> <ms_round> <ms_wait> <ms_draft> [wprob 0..n]
//
// `wprob` is present only when the MTP drafted for that window (16 entries, -1 where no estimate exists), so a line
// has either 7 fields or 23.  Extra fields appended later are ignored, which is what lets the format grow.
//
// The line does NOT carry the suffix match length, so a lookup window cannot be fed to Controller::observe exactly;
// a replayed sequence here is therefore MTP-only, and the lookup windows are skipped rather than approximated.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace basalt::spec {

struct DraftLogWindow {
    int T = 0;              ///< verify window width
    int accepted = 0;       ///< drafts the window accepted (0..T-1)
    bool from_sfx = false;  ///< the window's drafts came from the suffix lookup
    double ms_round = 0, ms_wait = 0, ms_draft = 0;
    bool has_probs = false; ///< the MTP drafted: `wprob` follows
};

/// Parses the `W` lines of a draft log, in file order.  Lines that are not `W` lines (the `R` header, comments) and
/// malformed `W` lines are skipped, and `skipped` counts them, so a truncated file replays what it has.
inline std::vector<DraftLogWindow> draft_log_windows(const std::string& text, size_t* skipped = nullptr) {
    std::vector<DraftLogWindow> out;
    size_t bad = 0;
    for (size_t at = 0; at < text.size();) {
        const size_t end = text.find('\n', at);
        const std::string line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        at = end == std::string::npos ? text.size() : end + 1;
        if (line.size() < 2 || line[0] != 'W' || line[1] != ' ') continue;   // not a window line
        // Fields: W and six values, then the optional probability block.
        const char* p = line.c_str() + 2;
        auto field = [&](const char* what, bool& ok) -> double {
            while (*p == ' ') ++p;
            if (*p == '\0') { ok = false; return 0.0; }
            char* stop = nullptr;
            const double v = std::strtod(p, &stop);
            if (stop == p) { ok = false; return 0.0; }
            p = stop;
            (void) what;
            return v;
        };
        bool ok = true;
        DraftLogWindow w;
        w.T = (int) field("T", ok);
        w.accepted = (int) field("accepted", ok);
        w.from_sfx = field("from_sfx", ok) != 0.0;
        w.ms_round = field("ms_round", ok);
        w.ms_wait = field("ms_wait", ok);
        w.ms_draft = field("ms_draft", ok);
        if (!ok || w.T < 1 || w.T > 64 || w.accepted < 0 || w.accepted >= w.T) { ++bad; continue; }
        w.has_probs = *p != '\0';
        out.push_back(w);
    }
    if (skipped != nullptr) *skipped = bad;
    return out;
}

}  // namespace basalt::spec
