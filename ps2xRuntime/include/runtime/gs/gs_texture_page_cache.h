#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace GSMem
{
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;
        static constexpr uint32_t kBlockSize = 256u;
        static constexpr uint32_t kSlotCount = 1024u;
        static constexpr uint32_t kSlotMask = kSlotCount - 1u;

        TexturePageCache() noexcept
        {
            m_slotGen.fill(0u);
            m_blockBases.fill(UINT32_MAX);
        }

        void Invalidate() noexcept
        {
            if (++m_generation == 0u)
            {
                m_slotGen.fill(0u);
                m_generation = 1u;
            }
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss on the same slot or invalidation.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t blockBase = byteAddress & ~(kBlockSize - 1u);
            const uint32_t slot = (blockBase >> 8u) & kSlotMask;
            if (m_slotGen[slot] != m_generation || m_blockBases[slot] != blockBase)
            {
                std::memcpy(m_bytes[slot].data(), vram + blockBase, kBlockSize);
                m_blockBases[slot] = blockBase;
                m_slotGen[slot] = m_generation;
            }
            return m_bytes[slot].data() + (byteAddress & (kBlockSize - 1u));
        }

    private:
        alignas(64) std::array<std::array<uint8_t, kBlockSize>, kSlotCount> m_bytes{};
        std::array<uint32_t, kSlotCount> m_blockBases{};
        std::array<uint32_t, kSlotCount> m_slotGen{};
        uint32_t m_generation = 1u;
    };
}
