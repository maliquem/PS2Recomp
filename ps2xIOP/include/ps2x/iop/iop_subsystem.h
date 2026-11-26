#pragma once

#include "ps2x/iop/iop_host.h"
#include "ps2x/iop/iop_types.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ps2x::state
{
    class Archive;
}

namespace ps2x::iop
{
    class IopSubsystem
    {
    public:
        explicit IopSubsystem(IopHost &host);
        ~IopSubsystem();

        IopSubsystem(const IopSubsystem &) = delete;
        IopSubsystem &operator=(const IopSubsystem &) = delete;
        IopSubsystem(IopSubsystem &&) noexcept;
        IopSubsystem &operator=(IopSubsystem &&) noexcept;

        void reset();

        [[nodiscard]] ModuleLoadResult loadModule(std::string_view path, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] bool stopModule(int32_t moduleId, int32_t *result = nullptr);
        void runEeCycles(uint64_t eeCycles) noexcept;
        [[nodiscard]] uint64_t idleEeCycleHorizon() const noexcept;

        [[nodiscard]] RpcAbi selectRpcAbi(const RpcAbiRequest &request) const;
        [[nodiscard]] bool canBindRpc(uint32_t sid) const noexcept;
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request);
        void onSifTransfer(const SifTransfer &transfer);

        // Physical IOP RAM access shared by the emulator, SIF DMA, and HLE services. Addresses are IOP addresses.
        [[nodiscard]] uint32_t allocateMemory(uint32_t size, uint32_t alignment = 16u);
        [[nodiscard]] bool freeMemory(uint32_t address);
        [[nodiscard]] bool readMemory(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeMemory(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroMemory(uint32_t address, size_t size);
        [[nodiscard]] bool isMemoryRange(uint32_t address, size_t size) const;

        [[nodiscard]] DebugSnapshot debugSnapshot() const;

        // Game profiles. A code hook runs before the instruction at `offset` of every loaded IRX whose
        // file contents hash to `imageHash` (64-bit FNV-1a; the load log prints it), so another build
        // of the module simply gets no hooks. The address must not be a branch delay slot. Hooks
        // survive reset() and run on the thread that executes the IOP.
        void addCodeHook(uint64_t imageHash, uint32_t offset, CodeHook hook);
        // SPU2 output buses (the numbers of ps2x::audio in runtime/ps2_audio_mix.h): the bus a voice
        // plays on from its next key-on and every one after it, until set again (so a profile either
        // tags every key-on or assigns voices once), the bus of a core's PCM input (kept across reset()), and
        // how many samples a bus has advanced (the music bus falls behind bus 0 while it runs slower).
        void setSpu2VoiceBus(int core, uint32_t voice, int bus);
        void setSpu2InputBus(int core, int bus);
        [[nodiscard]] uint32_t spu2BusSamples(int bus) const noexcept;

        // Save state: saves or restores the whole IOP (same executable only). Call only while the IOP is
        // not executing (outside runEeCycles and RPC handling).
        void stateIo(ps2x::state::Archive &ar);

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
