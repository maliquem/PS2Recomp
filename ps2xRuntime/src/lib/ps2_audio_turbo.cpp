// Turbo time compressor (runtime/ps2_audio_turbo.h). WSOLA ported from the BoF4 recomp: grains of kN
// frames under a periodic Hann window, one every kHs output frames (the two halves add up to 1). The
// next grain's input position is the last one's plus the analysis hop kHs x ratio, moved by up to
// kDelta frames to where it best matches the last grain's natural continuation. The ratio is the
// measured input over the output asked for, corrected by how far the held input is from what the next
// grain needs, so the added delay stays small (~60 ms at 2x).

#include "runtime/ps2_audio_turbo.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps2x::audio
{
    namespace
    {
        struct HannWindow
        {
            std::array<float, 1024> w{};
            HannWindow()
            {
                for (size_t i = 0; i < w.size(); ++i)
                    w[i] = 0.5f - 0.5f * std::cos(6.283185307f * static_cast<float>(i) / static_cast<float>(w.size()));
            }
        };
        const HannWindow g_win;

        int16_t clamp16(float v)
        {
            v *= 32768.0f;
            if (v > 32767.0f)
                return 32767;
            if (v < -32768.0f)
                return -32768;
            return static_cast<int16_t>(std::lrintf(v));
        }
    }

    void TurboCompressor::reset()
    {
        m_end = m_rd = m_prev = 0;
        m_compressing = false;
        m_finN = m_finOff = 0;
        m_ratio = 1.0;
        m_inSince = m_outSince = 0;
        for (auto &p : m_pend)
            p = {0.0f, 0.0f};
    }

    void TurboCompressor::push(const int16_t *in, size_t frames)
    {
        for (size_t f = 0; f < frames; ++f)
        {
            auto &d = m_in[static_cast<size_t>(m_end & kMask)];
            d[0] = in[f * 2u];
            d[1] = in[f * 2u + 1u];
            ++m_end;
        }
        m_inSince += static_cast<int64_t>(frames);
        // Direct: keep the old ring's 0.5 s latency cap (drops the oldest). Compressing: pull() jumps
        // ahead when the backlog grows too large.
        if (!m_compressing && m_end - m_rd > kDirectCap)
            m_rd = m_end - kDirectCap;
    }

    float TurboCompressor::corr(int64_t p, int64_t nat, int step) const
    {
        float xy = 0.0f, yy = 1e-9f;
        for (int i = 0; i < kLc; i += step)
        {
            const float x = mono(nat + i), y = mono(p + i);
            xy += x * y;
            yy += y * y;
        }
        return xy / std::sqrt(yy);
    }

    int64_t TurboCompressor::bestPos(int64_t lo, int64_t hi, int64_t nat) const
    {
        float best = -1e30f;
        int64_t bp = lo;
        for (int64_t p = lo; p <= hi; p += 4)
        {
            const float c = corr(p, nat, 2);
            if (c > best)
            {
                best = c;
                bp = p;
            }
        }
        const int64_t a = std::max(lo, bp - 3), b = std::min(hi, bp + 3);
        best = -1e30f;
        for (int64_t p = a; p <= b; ++p)
        {
            const float c = corr(p, nat, 1);
            if (c > best)
            {
                best = c;
                bp = p;
            }
        }
        return bp;
    }

    size_t TurboCompressor::takeFinished(int16_t *out, size_t room)
    {
        size_t n = static_cast<size_t>(std::max(0, m_finN - m_finOff));
        n = std::min(n, room);
        if (n == 0u)
            return 0u;
        std::memcpy(out, m_fin[static_cast<size_t>(m_finOff)].data(), n * 2u * sizeof(int16_t));
        m_finOff += static_cast<int>(n);
        return n;
    }

    void TurboCompressor::measure(size_t asked)
    {
        m_outSince += static_cast<int64_t>(asked);
        if (m_outSince < kMeasure)
            return;
        const double r = std::clamp(static_cast<double>(m_inSince) / static_cast<double>(m_outSince), 0.5, 64.0);
        m_ratio = 0.6 * m_ratio + 0.4 * r;
        m_inSince = m_outSince = 0;
    }

    size_t TurboCompressor::pull(int16_t *out, size_t want, bool compress)
    {
        const auto &win = g_win.w;
        size_t n = 0;
        if (!compress)
        {
            // Back to the input as it came: the hop in hand, then the last grain's natural continuation
            // (its pending falling half plus that input's rising half are the input itself).
            n = takeFinished(out, want);
            if (m_compressing && m_finOff >= m_finN)
            {
                m_compressing = false;
                const int64_t nat = m_prev + kHs;
                m_rd = nat;
                if (m_end - nat > 4 * kN)
                {
                    // Turbo left a backlog: jump near the newest input, crossfading out of the last grain
                    // at the best matching spot, so 1x play does not lag behind the game.
                    const int64_t target = m_end - 2 * kN;
                    const int64_t np = bestPos(target - kDelta, std::min<int64_t>(target + kDelta, m_end - kN), nat);
                    for (int i = 0; i < kHs; ++i)
                        for (int c = 0; c < 2; ++c)
                        {
                            const auto ii = static_cast<size_t>(i);
                            m_fin[ii][c] = clamp16(m_pend[ii][c] + sample(np + i, c) * win[ii]);
                        }
                    m_finN = kHs;
                    m_finOff = 0;
                    m_rd = np + kHs;
                    n += takeFinished(out + n * 2u, want - n);
                }
            }
            while (!m_compressing && n < want && m_rd < m_end)
            {
                const auto &d = m_in[static_cast<size_t>(m_rd & kMask)];
                out[n * 2u] = d[0];
                out[n * 2u + 1u] = d[1];
                ++m_rd;
                ++n;
            }
            measure(n);
            return n;
        }

        if (!m_compressing)
        {
            // Start as if a grain ended on what was just played: its falling half is the input from
            // here on, so the first grain overlaps in phase.
            if (m_end - m_rd < kHs)
                return takeFinished(out, want);
            n = takeFinished(out, want);
            for (int i = 0; i < kHs; ++i)
                for (int c = 0; c < 2; ++c)
                    m_pend[static_cast<size_t>(i)][c] = sample(m_rd + i, c) * win[static_cast<size_t>(kHs + i)];
            m_prev = m_rd - kHs;
            m_compressing = true;
        }
        else
        {
            n = takeFinished(out, want);
        }

        while (n < want)
        {
            // The pulls stalled and too much piled up: restart near the newest input.
            if (m_end - m_prev > kCap / 2)
                m_prev = m_end - static_cast<int64_t>(kHs * m_ratio) - kMargin;
            const int64_t nat = m_prev + kHs;
            const double target = static_cast<double>(kHs) * m_ratio + static_cast<double>(kMargin);
            double ha = static_cast<double>(kHs) * m_ratio + (static_cast<double>(m_end - m_prev) - target) * 0.25;
            ha = std::clamp(ha, 0.75 * kHs, 64.0 * kHs);
            int64_t p0 = m_prev + static_cast<int64_t>(ha);
            // The speed fell (the ratio still lags): take the grain from the input there is.
            if (p0 + kDelta + kN > m_end)
            {
                p0 = m_end - kDelta - kN;
                if (p0 < m_prev + static_cast<int64_t>(0.75 * kHs))
                    break; // wait for input
            }
            const int64_t lo = p0 - kDelta;
            const int64_t hi = std::min(p0 + kDelta, m_end - kN);
            if (hi < lo)
                break;
            const int64_t p = bestPos(lo, hi, nat);
            for (int i = 0; i < kHs; ++i)
            {
                const auto ii = static_cast<size_t>(i);
                for (int c = 0; c < 2; ++c)
                {
                    m_fin[ii][c] = clamp16(m_pend[ii][c] + sample(p + i, c) * win[ii]);
                    m_pend[ii][c] = sample(p + kHs + i, c) * win[static_cast<size_t>(kHs + i)];
                }
            }
            m_prev = p;
            m_finN = kHs;
            m_finOff = 0;
            n += takeFinished(out + n * 2u, want - n);
        }
        measure(want);
        return n;
    }
}
