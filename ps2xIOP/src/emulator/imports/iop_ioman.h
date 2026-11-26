#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ps2x::state
{
    class Archive;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopGuestExecutor;
    class IopMemory;

    class IopIoman
    {
    public:
        explicit IopIoman(IopMemory &memory) noexcept;

        void reset();
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor);

        // Save state: saves or restores the guest-visible state (same executable only).
        void stateIo(ps2x::state::Archive &ar);

    private:
        struct Device
        {
            uint32_t address = 0u;
            uint32_t gp = 0u;
            std::string name;

            template <class A>
            void stateIo(A &ar) { ar(address, gp, name); }
        };

        IopMemory &m_memory;
        std::vector<Device> m_devices;
    };
}
