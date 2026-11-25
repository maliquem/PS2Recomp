#include "runtime/gs/ps2_gif_arbiter.h"
#include <algorithm>
#include <cstring>

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    // FLG=3 ("disabled" in the GS manual) behaves exactly like IMAGE: NLOOP qwords of image data.
    return flg == 2u || flg == 3u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    const size_t offset = m_arena.size();
    m_arena.resize(offset + sizeBytes);
    std::memcpy(m_arena.data() + offset, data, sizeBytes);

    GifArbiterPacket pkt{};
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.offset = static_cast<uint32_t>(offset);
    pkt.size = sizeBytes;
    m_queue.push_back(pkt);
}

void GifArbiter::drain()
{
    if (!m_processFn || m_queue.empty())
        return;

    if (m_queue.size() > 1u)
    {
        std::stable_sort(m_queue.begin(), m_queue.end(),
                         [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                         {
                             // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                             if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                             {
                                 if (a.path3Image && b.path2DirectHl)
                                     return true;
                                 if (a.path2DirectHl && b.path3Image)
                                     return false;
                             }
                             return pathPriority(a.pathId) < pathPriority(b.pathId);
                         });
    }

    const uint8_t *base = m_arena.data();
    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        const auto &pkt = m_queue[i];
        if (pkt.size != 0u)
        {
            m_processFn(base + pkt.offset, pkt.size);
        }
    }
    m_queue.clear();
    m_arena.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
