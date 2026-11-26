// Save states, turbo and turbo frameskip.
//
// Machine state: every component has a symmetric stateIo(Archive&) (ps2x/state_archive.h). This file
// holds the ones for the runtime core (memory, VU, kernel tables) and the service that drives them:
// options from PS2X_* variables, hotkeys, slot files and the safe-point work on the EE thread.

#include "ps2_runtime.h"
#include "ps2_host_backend.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_pacing.h"
#include "runtime/ps2_savestate.h"
#include "runtime/ps2_vfs.h"
#include "runtime/ps2_vu1.h"
#include "ps2x/iop/iop_subsystem.h"
#include "ps2x/state_archive.h"
#include "Stubs/StubState.h"

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include "Syscalls/Helpers/State.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using ps2x::state::Archive;

namespace
{
    // Section marker: catches a stream that drifted out of step (a component that changed its layout).
    void section(Archive &ar, const char (&tag)[5], uint32_t version)
    {
        uint32_t fourcc = 0;
        std::memcpy(&fourcc, tag, 4);
        uint32_t readTag = fourcc;
        uint32_t readVersion = version;
        ar(readTag, readVersion);
        if (ar.loading() && (readTag != fourcc || readVersion != version))
            throw std::runtime_error(std::string("save state section mismatch at ") + tag);
    }
}

// ---------------------------------------------------------------------------------------------
// Runtime core state
// ---------------------------------------------------------------------------------------------

void PS2Memory::stateIo(Archive &ar)
{
    ar.raw(m_rdram, PS2_RAM_SIZE);
    ar.raw(m_scratchpad, PS2_SCRATCHPAD_SIZE);
    ar.raw(iop_ram, 2u * 1024u * 1024u);
    ar.raw(m_vu0Code, PS2_VU0_CODE_SIZE);
    ar.raw(m_vu0Data, PS2_VU0_DATA_SIZE);
    ar.raw(m_vu1Code, PS2_VU1_CODE_SIZE);
    ar.raw(m_vu1Data, PS2_VU1_DATA_SIZE);
    // GSRegisters holds atomics (vsyncTick, CSR); the struct is plain storage otherwise.
    ar.raw(&gs_regs, sizeof(gs_regs));
    ar(m_seenGifCopy, m_ioRegisters, vif0_regs, vif1_regs, dma_regs);
    uint64_t tlbCount = ar.size(m_tlbEntries.size());
    if (ar.loading())
        m_tlbEntries.resize(static_cast<size_t>(tlbCount));
    for (TLBEntry &e : m_tlbEntries)
        ar(e.vpn, e.pfn, e.mask, e.valid);
    ar(m_path3Masked, m_vif1PendingDirectQwc, m_vif1PendingPath2ImageQwc, m_vif1PendingPath2DirectHl,
       m_path3MaskedFifo, m_pendingGifTransfers, m_pendingVif0Transfers, m_pendingVif1Transfers);
    {
        std::lock_guard<std::mutex> lock(m_completedDmacMutex);
        ar(m_completedDmacCauses);
    }
    for (EeTimer &t : m_eeTimers)
        ar(t.count, t.mode, t.compare, t.hold, t.clockRemainder);
    ar(m_eeTimerPendingCycles, m_eeTimerFlushCycles);
    if (ar.loading())
    {
        // GIF STAT is located again on first use; VU programs changed under the decoded caches.
        m_gifStatRegister = nullptr;
        markVU0CodeModified();
        markVU1CodeModified();
    }
}

void VU1Interpreter::stateIo(Archive &ar)
{
    ar(m_state, m_flagPipeline, m_fdiv, m_efu, m_storePipeline, m_vfWritePipeline, m_viWritePipeline, m_xgkick,
       m_vfReady, m_viReady, m_vfLatestWrite, m_viLatestWrite, m_cycle, m_nextWriteSequence, m_efuResourceReady,
       m_workingClip, m_currentUpperInstruction, m_viBranchBackupValue, m_viBranchBackupReg, m_viBranchBackupValid,
       m_stopRequested, m_pendingHaltD, m_pendingHaltT);
    if (ar.loading())
        m_decodedCodeCacheValid = false;
}

void PS2Vfs::stateIo(Archive &ar)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // shortcut: an open guest file cannot be reopened at the same position after loading, so saving
    // waits until the game has closed its files (the retry window covers the usual short reads).
    if (!ar.loading() && !m_descriptors.empty())
        throw ps2x::state::StateBusy("a game file is open");
    if (ar.loading())
        m_descriptors.clear();
    ar(m_nextDescriptor);
}

namespace
{
    void ioSifModule(Archive &ar, SifModuleRecord &m)
    {
        ar(m.id, m.path, m.pathKey, m.refCount, m.loaded);
    }

    void syscallStateIo(Archive &ar)
    {
        {
            std::lock_guard<std::mutex> lock(g_rpc_mutex);
            ar(g_rpc_servers, g_rpc_clients, g_rpc_initialized, g_rpc_next_id, g_rpc_packet_index,
               g_rpc_server_index, g_rpc_active_queue);
        }
        {
            std::lock_guard<std::mutex> lock(g_bootmode_mutex);
            ar(g_bootmode_initialized, g_bootmode_pool_offset, g_bootmode_addresses);
        }
        {
            std::lock_guard<std::mutex> lock(g_tls_mutex);
            ar(g_tls_index);
        }
        {
            std::lock_guard<std::mutex> lock(g_osd_mutex);
            ar(g_osd_config_initialized, g_osd_config_raw, g_osd_config2_raw);
        }
        std::lock_guard<std::mutex> lock(g_sif_module_mutex);
        // Saved in id order and re-inserted, so the guest sees the same ids and lookups.
        std::vector<int32_t> ids;
        for (const auto &[id, rec] : g_sif_modules_by_id)
            ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        uint64_t count = ar.size(ids.size());
        if (ar.loading())
            g_sif_modules_by_id.clear();
        for (uint64_t i = 0; i < count; ++i)
        {
            SifModuleRecord rec{};
            if (!ar.loading())
                rec = g_sif_modules_by_id.at(ids[static_cast<size_t>(i)]);
            ioSifModule(ar, rec);
            if (ar.loading())
                g_sif_modules_by_id.emplace(rec.id, rec);
        }
        ar(g_sif_module_id_by_path, g_next_sif_module_id);
    }
}

bool PS2Vfs::hasOpenFiles() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return !m_descriptors.empty();
}

const char *PS2Runtime::stateBusy()
{
    if (m_vfs.hasOpenFiles())
        return "a game file is open";
    if (const char *why = ps2_stubs::libcStubBusy())
        return why;
    if (const char *why = ps2_stubs::mcStubBusy())
        return why;
    return ps2_stubs::mpegStubBusy();
}

void PS2Runtime::stateIo(Archive &ar)
{
    static_assert(std::is_trivially_copyable_v<R5900Context>);

    section(ar, "MEM ", 1);
    m_memory.stateIo(ar);
    section(ar, "GIF ", 1);
    m_gifArbiter.stateIo(ar);
    section(ar, "GS  ", 1);
    m_gs.stateIo(ar);
    section(ar, "VU  ", 1);
    m_vu0.stateIo(ar);
    m_vu1.stateIo(ar);
    section(ar, "EE  ", 1);
    ar(m_cpuContext);
    m_eeScheduler->stateIo(ar);
    {
        std::lock_guard<std::mutex> lock(m_eeKernelStateMutex);
        uint64_t handlerThreads = ar.size(m_eeExitHandlers.size());
        if (ar.loading())
        {
            m_eeExitHandlers.clear();
            for (uint64_t i = 0; i < handlerThreads; ++i)
            {
                int threadId = 0;
                uint64_t n = 0;
                ar(threadId, n);
                auto &list = m_eeExitHandlers[threadId];
                list.resize(static_cast<size_t>(n));
                for (auto &h : list)
                    ar(h.function, h.argument);
            }
        }
        else
        {
            for (auto &[threadId, list] : m_eeExitHandlers)
            {
                int id = threadId;
                uint64_t n = list.size();
                ar(id, n);
                for (auto &h : list)
                    ar(h.function, h.argument);
            }
        }
        ar(m_eeSyscallOverrides, m_eeSyscallMirrorAddresses);
    }
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        uint64_t blocks = ar.size(m_guestHeapBlocks.size());
        if (ar.loading())
            m_guestHeapBlocks.resize(static_cast<size_t>(blocks));
        for (auto &b : m_guestHeapBlocks)
            ar(b.addr, b.size, b.free);
        ar(m_guestHeapBase, m_guestHeapEnd, m_guestHeapLimit, m_guestHeapSuggestedBase, m_guestHeapConfigured);
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        ar(m_asyncCallbackStackFloor, m_asyncCallbackStackTop);
    }
    uint64_t modules = ar.size(m_loadedModules.size());
    if (ar.loading())
        m_loadedModules.resize(static_cast<size_t>(modules));
    for (auto &m : m_loadedModules)
        ar(m.name, m.baseAddress, m.size, m.active);

    section(ar, "IOP ", 1);
    m_iopSubsystem->stateIo(ar);
    section(ar, "SYSC", 1);
    syscallStateIo(ar);
    section(ar, "STUB", 2);
    ps2_stubs::audioStubStateIo(ar);
    ps2_stubs::cdStubStateIo(ar);
    ps2_stubs::dmaStubStateIo(ar);
    ps2_stubs::gsStubStateIo(ar);
    ps2_stubs::libcStubStateIo(ar);
    ps2_stubs::mcStubStateIo(ar);
    ps2_stubs::mpegStubStateIo(ar);
    ps2_stubs::padStubStateIo(ar);
    ps2_stubs::sifStubStateIo(ar);
    m_vfs.stateIo(ar);
    section(ar, "END ", 1);

    if (ar.loading())
    {
        // Sound already queued belongs to the abandoned timeline.
        m_audioBackend.stopAll();
    }
}

// ---------------------------------------------------------------------------------------------
// Service: options, slots, hotkeys, safe-point work, turbo and frameskip
// ---------------------------------------------------------------------------------------------

namespace ps2x::savestate
{
    namespace
    {
        constexpr char kMagic[8] = {'P', 'S', '2', 'X', 'S', 'T', 'A', 'T'};
        constexpr uint32_t kFormatVersion = 1u;
        constexpr int kSlotCount = 10;
        constexpr auto kQuiescenceTimeout = std::chrono::seconds(2);
        constexpr auto kMessageTime = std::chrono::seconds(3);

        enum class Frameskip : int
        {
            Off = 0,
            Present = 1,  // drop every other present (GS draws everything)
            Raster = 2,   // GS skips drawing frames that are never shown
            Both = 3,     // 2 + 1
            Auto = 4,
        };

#pragma pack(push, 1)
        struct FileHeader
        {
            char magic[8];
            uint32_t formatVersion;
            uint32_t headerSize;
            char gameId[16];
            uint64_t exeHash;
            uint64_t payloadSize;
            uint64_t payloadHash;
            uint64_t vsyncTick;
        };
#pragma pack(pop)

        struct Options
        {
            std::filesystem::path dir;
            int slot = 1;
            uint32_t turboPercent = 300u; // 0 = unlimited
            Frameskip frameskip = Frameskip::Off;
            std::string loadAtStart;
            uint64_t saveAtField = 0;
            uint64_t selftestField = 0;
            uint64_t selftestFields = 120;
            std::filesystem::path hashLog;
            uint64_t hashFields = 300;
            bool turboAtStart = false;
            bool padHotkeys = true; // PS2X_PAD_HOTKEYS=0 frees L3 combos for games that use them
            uint64_t turboFrom = 0, turboTo = 0; // turbo on for fields [from, to) (tests)
            bool exitAfterTest = true;
        };

        enum class Op : int
        {
            None = 0,
            Save,
            Load,
        };

        // Guest state hash split by component, so a failed determinism test says what diverged.
        struct MachineHash
        {
            static constexpr size_t kParts = 10;
            static constexpr const char *kNames[kParts] = {"rdram", "scratchpad", "iop_ram", "vu_mem", "gs_regs",
                                                           "mem_other", "gs", "vu0", "vu1", "cpu"};
            uint64_t part[kParts]{};
            uint64_t total = 0;
        };

        struct Service
        {
            Options opt;
            std::string gameId;

            // Main thread -> EE.
            std::atomic<int> request{0}; // op | slot << 8, one atomic so the EE never sees a stale slot
            std::chrono::steady_clock::time_point requestSince{};
            bool requestWaiting = false;

            // Turbo / frameskip state (main thread writes, EE reads).
            std::atomic<bool> turboToggle{false};
            std::atomic<bool> turboHeld{false};
            std::atomic<int> frameskip{0};
            std::atomic<int> autoLevel{0};
            AutoFrameskip autoState; // main thread
            std::atomic<bool> vpsRestart{false}; // a load moved the field counter: restart the VPS sample
            std::atomic<bool> gpuPresent{false};
            uint64_t hostFrame = 0;

            // EE side.
            uint64_t lastTick = ~0ull;
            uint64_t requestTriedTick = ~0ull; // a waiting request is tried once per field
            bool startLoadDone = false;
            bool savedAt = false;
            std::vector<uint8_t> selftestState;
            std::vector<MachineHash> selftestHashes[2];
            int selftestPhase = 0; // 0 idle, 1 recording first run, 2 recording replay, 3 done
            uint64_t selftestBase = 0;
            std::ofstream hashOut;
            uint64_t hashFrom = 0, hashTo = 0;
            std::atomic<bool> testTurbo{false}; // EE writes, main thread reads

            // Messages for the window title (EE and main thread).
            std::mutex messageMutex;
            std::string message;
            bool messageNew = false;
            std::chrono::steady_clock::time_point messageUntil{};

            // Pad combo edges.
            bool padPrev[GAMEPAD_BUTTON_COUNT]{};

            // VPS for Auto frameskip.
            std::chrono::steady_clock::time_point vpsSince{};
            uint64_t vpsTicks = 0;
        };

        Service &svc()
        {
            static Service s;
            return s;
        }

        uint64_t hashBytes(const uint8_t *p, size_t n, uint64_t h = 0x9E3779B97F4A7C15ull)
        {
            size_t i = 0;
            for (; i + 8 <= n; i += 8)
            {
                uint64_t v;
                std::memcpy(&v, p + i, 8);
                h = (h ^ v) * 0xFF51AFD7ED558CCDull;
                h ^= h >> 32;
            }
            for (; i < n; ++i)
                h = (h ^ p[i]) * 0x100000001B3ull;
            h ^= h >> 29;
            return h * 0xC4CEB9FE1A85EC53ull;
        }

        std::filesystem::path exePath()
        {
#if defined(_WIN32)
            std::wstring buf(32768, L'\0');
            const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
            buf.resize(n);
            return std::filesystem::path(buf);
#else
            std::error_code ec;
            return std::filesystem::read_symlink("/proc/self/exe", ec);
#endif
        }

        uint64_t computeExeHash()
        {
            std::ifstream in(exePath(), std::ios::binary);
            if (!in)
                return 0;
            std::vector<uint8_t> buf(1u << 22);
            uint64_t h = 0x9E3779B97F4A7C15ull;
            while (in)
            {
                in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
                const auto got = static_cast<size_t>(in.gcount());
                if (got == 0)
                    break;
                h = hashBytes(buf.data(), got, h);
            }
            return h;
        }

        // Read on the first save or load, not at boot: players who never use save states pay nothing.
        uint64_t exeHash()
        {
            static const uint64_t hash = computeExeHash();
            if (hash == 0)
                throw std::runtime_error("cannot read the game executable to tag the save state");
            return hash;
        }

        void say(const std::string &text)
        {
            Service &s = svc();
            std::cout << "[savestate] " << text << std::endl;
            std::lock_guard<std::mutex> lock(s.messageMutex);
            s.message = text;
            s.messageNew = true;
        }

        uint64_t envU64(const char *name, uint64_t fallback)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return fallback;
            char *end = nullptr;
            const uint64_t value = std::strtoull(v, &end, 10);
            return (end == v || *end != '\0') ? fallback : value;
        }

        Frameskip parseFrameskip(const char *v)
        {
            if (!v || !*v)
                return Frameskip::Off;
            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (s == "1")
                return Frameskip::Present;
            if (s == "2")
                return Frameskip::Raster;
            if (s == "3")
                return Frameskip::Both;
            if (s == "auto")
                return Frameskip::Auto;
            return Frameskip::Off;
        }

        const char *frameskipName(int level)
        {
            switch (static_cast<Frameskip>(level))
            {
            case Frameskip::Present:
                return "1";
            case Frameskip::Raster:
                return "2";
            case Frameskip::Both:
                return "3";
            case Frameskip::Auto:
                return "Auto";
            default:
                return "Off";
            }
        }

        Options readOptions()
        {
            Options o;
            if (const char *d = std::getenv("PS2X_SAVESTATE_DIR"); d && *d)
                o.dir = d;
            else
            {
                const auto &paths = PS2Runtime::getIoPaths();
                o.dir = (paths.mcRoot.empty() ? std::filesystem::current_path() : paths.mcRoot.parent_path()) / "savestates";
            }
            o.slot = static_cast<int>(std::min<uint64_t>(envU64("PS2X_SAVESTATE_SLOT", 1), kSlotCount - 1));
            o.turboPercent = static_cast<uint32_t>(envU64("PS2X_TURBO_SPEED", 300));
            if (o.turboPercent != 0u)
                o.turboPercent = std::clamp<uint32_t>(o.turboPercent, 110u, 2000u);
            o.frameskip = parseFrameskip(std::getenv("PS2X_FRAMESKIP"));
            if (const char *l = std::getenv("PS2X_SAVESTATE_LOAD"); l && *l)
                o.loadAtStart = l;
            o.saveAtField = envU64("PS2X_SAVESTATE_SAVE_AT", 0);
            if (const char *t = std::getenv("PS2X_SAVESTATE_SELFTEST"); t && *t)
            {
                o.selftestField = std::strtoull(t, nullptr, 10);
                if (const char *comma = std::strchr(t, ','))
                    o.selftestFields = std::max<uint64_t>(1, std::strtoull(comma + 1, nullptr, 10));
            }
            if (const char *h = std::getenv("PS2X_SAVESTATE_HASH_LOG"); h && *h)
                o.hashLog = h;
            o.hashFields = envU64("PS2X_SAVESTATE_HASH_FIELDS", 300);
            o.turboAtStart = envU64("PS2X_TURBO", 0) != 0;
            o.padHotkeys = envU64("PS2X_PAD_HOTKEYS", 1) != 0;
            if (const char *r = std::getenv("PS2X_TURBO_FIELDS"); r && *r)
            {
                o.turboFrom = std::strtoull(r, nullptr, 10);
                if (const char *comma = std::strchr(r, ','))
                    o.turboTo = std::strtoull(comma + 1, nullptr, 10);
            }
            o.exitAfterTest = envU64("PS2X_SAVESTATE_TEST_EXIT", 1) != 0;
            return o;
        }

        std::filesystem::path slotPath(int slot)
        {
            Service &s = svc();
            return s.opt.dir / (s.gameId + ".slot" + std::to_string(slot) + ".ps2xstate");
        }

        bool turboActive()
        {
            Service &s = svc();
            return s.turboToggle.load(std::memory_order_relaxed) || s.turboHeld.load(std::memory_order_relaxed) || s.testTurbo;
        }

        void applySpeed()
        {
            Service &s = svc();
            ps2x::pacing::g_speedPercent.store(turboActive() ? s.opt.turboPercent : 100u, std::memory_order_relaxed);
        }

        int effectiveFrameskip()
        {
            Service &s = svc();
            if (!turboActive())
                return 0;
            int level = s.frameskip.load(std::memory_order_relaxed);
            if (level == static_cast<int>(Frameskip::Auto))
                level = s.autoLevel.load(std::memory_order_relaxed);
            return level;
        }

        // ---- machine snapshot ----------------------------------------------------------------

        std::vector<uint8_t> captureMachine(PS2Runtime &rt)
        {
            std::vector<uint8_t> payload;
            payload.reserve(48u << 20);
            Archive ar(payload);
            rt.stateIo(ar);
            return payload;
        }

        void restoreMachine(PS2Runtime &rt, const std::vector<uint8_t> &payload)
        {
            Archive ar(payload.data(), payload.size());
            rt.stateIo(ar);
            if (ar.remaining() != 0)
                throw std::runtime_error("save state has trailing data");
        }

        // Hash of the deterministic guest state (no host deadlines), for the determinism tests.
        MachineHash machineHash(PS2Runtime &rt)
        {
            std::vector<uint8_t> bytes;
            bytes.reserve(48u << 20);
            Archive ar(bytes);
            size_t ends[MachineHash::kParts]{};
            // PS2Memory::stateIo writes these raw blocks first, in this order.
            ends[0] = PS2_RAM_SIZE;
            ends[1] = ends[0] + PS2_SCRATCHPAD_SIZE;
            ends[2] = ends[1] + 2u * 1024u * 1024u;
            ends[3] = ends[2] + PS2_VU0_CODE_SIZE + PS2_VU0_DATA_SIZE + PS2_VU1_CODE_SIZE + PS2_VU1_DATA_SIZE;
            ends[4] = ends[3] + sizeof(GSRegisters);
            rt.memory().stateIo(ar);
            ends[5] = bytes.size();
            rt.gs().stateIo(ar);
            ends[6] = bytes.size();
            rt.vu0().stateIo(ar);
            ends[7] = bytes.size();
            rt.vu1().stateIo(ar);
            ends[8] = bytes.size();
            ar(rt.cpu());
            ends[9] = bytes.size();
            MachineHash h;
            size_t from = 0;
            for (size_t i = 0; i < MachineHash::kParts; ++i)
            {
                h.part[i] = hashBytes(bytes.data() + from, ends[i] - from);
                from = ends[i];
            }
            h.total = hashBytes(bytes.data(), bytes.size());
            return h;
        }

        std::string differingParts(const MachineHash &a, const MachineHash &b)
        {
            std::string out;
            for (size_t i = 0; i < MachineHash::kParts; ++i)
                if (a.part[i] != b.part[i])
                    out += std::string(out.empty() ? "" : ",") + MachineHash::kNames[i];
            return out.empty() ? "none" : out;
        }

        void writeFile(const std::filesystem::path &path, const std::vector<uint8_t> &payload, uint64_t tick)
        {
            writeStateFile(path, payload, svc().gameId, exeHash(), tick);
        }

        std::vector<uint8_t> readFile(const std::filesystem::path &path)
        {
            return readStateFile(path, svc().gameId, exeHash());
        }

        std::filesystem::path resolveLoadTarget(const std::string &v)
        {
            if (v.size() == 1 && v[0] >= '0' && v[0] <= '9')
                return slotPath(v[0] - '0');
            return std::filesystem::path(v);
        }

        // ---- EE thread ------------------------------------------------------------------------

        void resetFieldTracking(PS2Runtime &rt)
        {
            Service &s = svc();
            s.lastTick = rt.eeScheduler().currentVSyncTick();
        }

        // Load: keep a copy of the running machine so a bad file never leaves it half-loaded.
        void loadPayload(PS2Runtime &rt, const std::vector<uint8_t> &payload)
        {
            std::vector<uint8_t> backup = captureMachine(rt);
            try
            {
                restoreMachine(rt, payload);
            }
            catch (...)
            {
                restoreMachine(rt, backup);
                throw;
            }
            resetFieldTracking(rt);
        }

        bool quiescent(PS2Runtime &rt)
        {
            return rt.eeScheduler().stateQuiescent();
        }

        // Runs a request; false = not at a usable point yet (retry on a later safe point).
        bool runRequest(PS2Runtime &rt, Op op, int slot)
        {
            const uint64_t tick = rt.eeScheduler().currentVSyncTick();
            try
            {
                // Cheap checks first: the capture copies ~40 MB and waits for the GPU.
                if (!quiescent(rt) || (op == Op::Save && rt.stateBusy()))
                    return false;
                if (op == Op::Save)
                {
                    std::vector<uint8_t> payload = captureMachine(rt);
                    writeFile(slotPath(slot), payload, tick);
                    say("Saved state to slot " + std::to_string(slot));
                }
                else
                {
                    loadPayload(rt, readFile(slotPath(slot)));
                    svc().vpsRestart.store(true, std::memory_order_relaxed);
                    say("Loaded state from slot " + std::to_string(slot));
                }
            }
            catch (const ps2x::state::StateBusy &)
            {
                return false; // a file or movie in flight: wait for a better moment inside the retry window
            }
            catch (const std::exception &e)
            {
                say(std::string(op == Op::Save ? "Save" : "Load") + " failed: " + e.what());
            }
            return true;
        }

        void handleRequest(PS2Runtime &rt)
        {
            Service &s = svc();
            const int packed = s.request.load(std::memory_order_acquire);
            const int op = packed & 0xFF;
            if (op == 0)
                return;
            const auto now = std::chrono::steady_clock::now();
            const uint64_t tick = rt.eeScheduler().currentVSyncTick();
            if (!s.requestWaiting)
            {
                s.requestWaiting = true;
                s.requestSince = now;
            }
            else if (tick == s.requestTriedTick && now - s.requestSince < kQuiescenceTimeout)
                return;
            s.requestTriedTick = tick;
            if (!runRequest(rt, static_cast<Op>(op), packed >> 8))
            {
                if (now - s.requestSince < kQuiescenceTimeout)
                    return;
                const char *why = op == static_cast<int>(Op::Save) ? rt.stateBusy() : nullptr; // its reasons only block saves
                say(std::string(op == static_cast<int>(Op::Save) ? "Save" : "Load") + " failed: the game is busy (" +
                    (why ? why : "inside a system call") + "), try again");
            }
            s.requestWaiting = false;
            s.request.store(0, std::memory_order_release);
        }

        // Save refused because a file or movie is in flight (a later field may work).
        bool busyError(const std::exception &e)
        {
            return dynamic_cast<const ps2x::state::StateBusy *>(&e) != nullptr;
        }

        void finishTest(PS2Runtime &rt)
        {
            if (svc().opt.exitAfterTest)
                rt.requestStop();
        }

        void onField(PS2Runtime &rt, uint64_t tick)
        {
            Service &s = svc();
            const Options &o = s.opt;

            if (o.turboTo > o.turboFrom)
            {
                const bool on = tick >= o.turboFrom && tick < o.turboTo;
                if (on != s.testTurbo)
                {
                    s.testTurbo = on;
                    applySpeed();
                    std::cout << "[savestate] test turbo " << (on ? "on" : "off") << " at field " << tick << std::endl;
                }
            }

            // Determinism log (cross-process check): hashes of the fields after a save or a load.
            if (s.hashOut.is_open() && tick > s.hashFrom && tick <= s.hashTo)
            {
                const MachineHash h = machineHash(rt);
                s.hashOut << tick << ' ' << std::hex << h.total;
                for (uint64_t part : h.part)
                    s.hashOut << ' ' << part;
                s.hashOut << std::dec << '\n';
                if (tick == s.hashTo)
                {
                    s.hashOut.close();
                    std::cout << "[savestate] hash log done at field " << tick << std::endl;
                    finishTest(rt);
                }
            }

            if (o.saveAtField != 0 && !s.savedAt && tick >= o.saveAtField)
            {
                if (!quiescent(rt) || rt.stateBusy())
                    return;
                try
                {
                    writeFile(slotPath(o.slot), captureMachine(rt), tick);
                    s.savedAt = true;
                    say("Saved state to slot " + std::to_string(o.slot) + " at field " + std::to_string(tick));
                    if (!o.hashLog.empty())
                    {
                        s.hashOut.open(o.hashLog, std::ios::trunc);
                        s.hashFrom = tick;
                        s.hashTo = tick + o.hashFields;
                    }
                }
                catch (const std::exception &e)
                {
                    // A file or movie in flight: try again on a later field.
                    if (!busyError(e))
                    {
                        s.savedAt = true;
                        say(std::string("Save failed: ") + e.what());
                    }
                }
            }

            // In-process determinism self test: save, record, load, record again, compare.
            if (o.selftestField != 0)
            try
            {
                if (s.selftestPhase == 0 && tick >= o.selftestField && quiescent(rt) && !rt.stateBusy())
                {
                    s.selftestState = captureMachine(rt);
                    s.selftestBase = tick;
                    s.selftestPhase = 1;
                    std::cout << "[savestate] selftest: saved at field " << tick << std::endl;
                }
                else if (s.selftestPhase == 1 || s.selftestPhase == 2)
                {
                    auto &hashes = s.selftestHashes[s.selftestPhase - 1];
                    hashes.push_back(machineHash(rt));
                    if (hashes.size() == o.selftestFields)
                    {
                        if (s.selftestPhase == 1)
                        {
                            s.selftestPhase = 2;
                            loadPayload(rt, s.selftestState);
                            std::cout << "[savestate] selftest: loaded back to field " << rt.eeScheduler().currentVSyncTick() << std::endl;
                        }
                        else
                        {
                            s.selftestPhase = 3;
                            size_t firstBad = hashes.size();
                            for (size_t i = 0; i < hashes.size(); ++i)
                                if (s.selftestHashes[0][i].total != s.selftestHashes[1][i].total)
                                {
                                    firstBad = i;
                                    break;
                                }
                            if (firstBad == hashes.size())
                                std::cout << "[savestate] selftest PASS: " << hashes.size() << " fields identical after load" << std::endl;
                            else
                                std::cout << "[savestate] selftest FAIL: first difference " << (firstBad + 1)
                                          << " fields after the save (field " << (s.selftestBase + firstBad + 1) << "), differs in "
                                          << differingParts(s.selftestHashes[0][firstBad], s.selftestHashes[1][firstBad]) << std::endl;
                            finishTest(rt);
                        }
                    }
                }
            }
            catch (const std::exception &e)
            {
                if (s.selftestPhase == 0 && busyError(e))
                    return; // try again on a later field
                s.selftestPhase = 3;
                std::cout << "[savestate] selftest FAIL: " << e.what() << std::endl;
                finishTest(rt);
            }
        }

        void onSafePointUnguarded(PS2Runtime &rt)
        {
            Service &s = svc();
            if (!s.startLoadDone)
            {
                if (s.opt.loadAtStart.empty())
                    s.startLoadDone = true;
                else if (quiescent(rt))
                {
                    s.startLoadDone = true;
                    try
                    {
                        loadPayload(rt, readFile(resolveLoadTarget(s.opt.loadAtStart)));
                        s.vpsRestart.store(true, std::memory_order_relaxed);
                        say("Loaded state " + s.opt.loadAtStart + " (field " + std::to_string(rt.eeScheduler().currentVSyncTick()) + ")");
                        if (!s.opt.hashLog.empty())
                        {
                            s.hashOut.open(s.opt.hashLog, std::ios::trunc);
                            s.hashFrom = rt.eeScheduler().currentVSyncTick();
                            s.hashTo = s.hashFrom + s.opt.hashFields;
                        }
                    }
                    catch (const std::exception &e)
                    {
                        say(std::string("Load failed: ") + e.what());
                    }
                }
            }

            const uint64_t tick = rt.eeScheduler().currentVSyncTick();
            if (tick != s.lastTick)
            {
                s.lastTick = tick;
                onField(rt, tick);
            }

            // Frameskip level 2+: the GS reads this at each buffer swap.
            const bool rasterSkip = s.gpuPresent.load(std::memory_order_relaxed) && effectiveFrameskip() >= 2;
            ps2x::pacing::g_frameskipDraws.store(rasterSkip, std::memory_order_relaxed);

            handleRequest(rt);
        }

        void onSafePoint(PS2Runtime &rt)
        {
            try
            {
                onSafePointUnguarded(rt);
            }
            catch (const std::exception &e)
            {
                say(std::string("Save state error: ") + e.what());
            }
        }

        // ---- main thread ----------------------------------------------------------------------

        bool padDown(int button)
        {
#if PS2X_HOST_RAYLIB
            return IsGamepadAvailable(0) && IsGamepadButtonDown(0, button);
#else
            // SDL host: port 0 after the bindings (libpad bits), neutral while the pause menu is open.
            uint16_t bit = 0;
            switch (button)
            {
            case GAMEPAD_BUTTON_LEFT_THUMB: bit = 0x0002; break;
            case GAMEPAD_BUTTON_RIGHT_THUMB: bit = 0x0004; break;
            case GAMEPAD_BUTTON_LEFT_FACE_RIGHT: bit = 0x0020; break;
            case GAMEPAD_BUTTON_LEFT_FACE_LEFT: bit = 0x0080; break;
            case GAMEPAD_BUTTON_LEFT_TRIGGER_1: bit = 0x0400; break;
            case GAMEPAD_BUTTON_RIGHT_TRIGGER_1: bit = 0x0800; break;
            default: break;
            }
            return (ps2host::padInput(0).held & bit) != 0;
#endif
        }

        bool padPressed(int button)
        {
            Service &s = svc();
            const bool down = padDown(button);
            const bool pressed = down && !s.padPrev[button];
            s.padPrev[button] = down;
            return pressed;
        }

        void request(Op op, int slot)
        {
            Service &s = svc();
            int expected = 0;
            s.request.compare_exchange_strong(expected, static_cast<int>(op) | (slot << 8), std::memory_order_acq_rel); // one at a time
        }
    }

    int autoFrameskipStep(AutoFrameskip &a, double vps, double target)
    {
        const int l = std::clamp(a.level, 0, 3);
        for (int &age : a.age)
            age = std::min(age + 1, kAutoStaleSeconds + 1);
        const auto fresh = [&](int i) { return a.age[i] <= kAutoStaleSeconds; };
        a.vps[l] = fresh(l) ? (a.vps[l] + vps) / 2.0 : vps; // one-second samples swing by ~10%
        a.age[l] = 0;
        const double enough = target * 0.95;
        if (vps < enough)
        {
            if (l < 3 && !fresh(l + 1))
                return a.level = l + 1;
            // A lower level is measured again only if it was close: probing slower levels costs speed.
            if (l > 0 && !fresh(l - 1) && a.vps[l - 1] >= a.vps[l] * 0.97)
                return a.level = l - 1;
            double fastest = 0.0;
            for (int i = 0; i < 4; ++i)
                if (fresh(i))
                    fastest = std::max(fastest, a.vps[i]);
            int best = 0; // the lowest fresh level within 2% of the fastest: more frames shown for the same speed
            while (!fresh(best) || a.vps[best] * 1.02 < fastest)
                ++best;
            return a.level = best;
        }
        if (l > 0 && (!fresh(l - 1) || a.vps[l - 1] >= enough))
            return a.level = l - 1;
        return a.level = l;
    }

    void writeStateFile(const std::filesystem::path &path, const std::vector<uint8_t> &payload,
                        const std::string &gameId, uint64_t exeHash, uint64_t vsyncTick)
    {
        FileHeader h{};
        std::memcpy(h.magic, kMagic, sizeof(kMagic));
        h.formatVersion = kFormatVersion;
        h.headerSize = sizeof(FileHeader);
        std::strncpy(h.gameId, gameId.c_str(), sizeof(h.gameId) - 1);
        h.exeHash = exeHash;
        h.payloadSize = payload.size();
        h.payloadHash = hashBytes(payload.data(), payload.size());
        h.vsyncTick = vsyncTick;

        std::filesystem::create_directories(path.parent_path());
        std::filesystem::path tmp = path;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(&h), sizeof(h));
            out.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            out.close(); // the last bytes are written here: never rename a cut-off file over a good slot
            if (!out)
            {
                std::error_code ec;
                std::filesystem::remove(tmp, ec);
                throw std::runtime_error("cannot write " + tmp.string());
            }
        }
        std::filesystem::rename(tmp, path);
    }

    std::vector<uint8_t> readStateFile(const std::filesystem::path &path, const std::string &gameId, uint64_t exeHash)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("no save state in this slot");
        FileHeader h{};
        in.read(reinterpret_cast<char *>(&h), sizeof(h));
        if (!in || std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0)
            throw std::runtime_error("not a save state file");
        if (h.formatVersion != kFormatVersion || h.headerSize != sizeof(FileHeader))
            throw std::runtime_error("save state format " + std::to_string(h.formatVersion) + " is not supported");
        h.gameId[sizeof(h.gameId) - 1] = '\0';
        if (gameId != h.gameId)
            throw std::runtime_error(std::string("save state is for another game (") + h.gameId + ")");
        if (h.exeHash != exeHash)
            throw std::runtime_error("save state was made by a different build of the game");
        if (h.payloadSize > (512ull << 20))
            throw std::runtime_error("save state corrupt (size)");
        std::vector<uint8_t> payload(static_cast<size_t>(h.payloadSize));
        in.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!in || hashBytes(payload.data(), payload.size()) != h.payloadHash)
            throw std::runtime_error("save state corrupt (checksum)");
        return payload;
    }

    void install(PS2Runtime &runtime)
    {
        Service &s = svc();
        s.opt = readOptions();
        s.gameId = PS2Runtime::getIoPaths().elfPath.filename().string();
        std::transform(s.gameId.begin(), s.gameId.end(), s.gameId.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        s.frameskip.store(static_cast<int>(s.opt.frameskip), std::memory_order_relaxed);
        s.turboToggle.store(s.opt.turboAtStart, std::memory_order_relaxed);
#if !PS2X_HOST_RAYLIB
        s.gpuPresent.store(ps2host::gpuPresent(), std::memory_order_relaxed);
#endif
        applySpeed();
        runtime.eeScheduler().setSafePointHook([&runtime]() { onSafePoint(runtime); });
    }

    void onHostFrame(PS2Runtime &runtime)
    {
        Service &s = svc();
        ++s.hostFrame;
        const auto now = std::chrono::steady_clock::now();

        int slot = s.opt.slot;
        // Pad hotkeys all need L3 held, so plain game input never triggers them.
        const bool comboHeld = s.opt.padHotkeys && padDown(GAMEPAD_BUTTON_LEFT_THUMB);
        const bool padSave = padPressed(GAMEPAD_BUTTON_RIGHT_TRIGGER_1) && comboHeld;
        const bool padLoad = padPressed(GAMEPAD_BUTTON_LEFT_TRIGGER_1) && comboHeld;
        const bool padPrev = padPressed(GAMEPAD_BUTTON_LEFT_FACE_LEFT) && comboHeld;
        const bool padNext = padPressed(GAMEPAD_BUTTON_LEFT_FACE_RIGHT) && comboHeld;

        if (IsKeyPressed(KEY_F6) || padPrev)
        {
            s.opt.slot = (slot + kSlotCount - 1) % kSlotCount;
            say("Slot " + std::to_string(s.opt.slot));
        }
        if (IsKeyPressed(KEY_F8) || padNext)
        {
            s.opt.slot = (slot + 1) % kSlotCount;
            say("Slot " + std::to_string(s.opt.slot));
        }
        if (IsKeyPressed(KEY_F5) || padSave)
            request(Op::Save, s.opt.slot);
        if (IsKeyPressed(KEY_F7) || padLoad)
            request(Op::Load, s.opt.slot);

        // Turbo: F9 toggles, ` (grave) or L3+R3 hold.
        const bool wasTurbo = turboActive();
        if (IsKeyPressed(KEY_F9))
            s.turboToggle.store(!s.turboToggle.load(std::memory_order_relaxed), std::memory_order_relaxed);
        s.turboHeld.store(IsKeyDown(KEY_GRAVE) || (comboHeld && padDown(GAMEPAD_BUTTON_RIGHT_THUMB)),
                          std::memory_order_relaxed);
        if (IsKeyPressed(KEY_F10))
        {
            const int next = (s.frameskip.load(std::memory_order_relaxed) + 1) % 5;
            s.frameskip.store(next, std::memory_order_relaxed);
            s.autoState = {}; // a sample run at the old level must not count for Auto
            s.autoLevel.store(0, std::memory_order_relaxed);
            s.vpsRestart.store(true, std::memory_order_relaxed);
            say(std::string("Turbo frameskip: ") + frameskipName(next));
        }
        const bool isTurbo = turboActive();
        if (isTurbo != wasTurbo || s.vpsRestart.exchange(false, std::memory_order_relaxed))
        {
            s.vpsSince = now;
            s.vpsTicks = runtime.memory().gs().vsyncTick.load(std::memory_order_acquire);
        }
        if (isTurbo != wasTurbo)
        {
            applySpeed();
            s.autoState = {};
            s.autoLevel.store(0, std::memory_order_relaxed);
            if (s.turboToggle.load(std::memory_order_relaxed) || !s.turboHeld.load(std::memory_order_relaxed))
                say(isTurbo ? (s.opt.turboPercent ? "Turbo " + std::to_string(s.opt.turboPercent / 100.0).substr(0, 3) + "x" : std::string("Turbo unlimited"))
                            : std::string("Turbo off"));
        }

        // Once a second in turbo: Auto frameskip picks the level for the next second, and the speed and the
        // dropped draws go to stdout.
        if (isTurbo && now - s.vpsSince >= std::chrono::seconds(1))
        {
            const uint64_t ticks = runtime.memory().gs().vsyncTick.load(std::memory_order_acquire);
            const double seconds = std::chrono::duration<double>(now - s.vpsSince).count();
            const double vps = static_cast<double>(ticks - s.vpsTicks) / seconds;
            const double target = s.opt.turboPercent ? 59.94 * s.opt.turboPercent / 100.0 : 1e9;
            if (s.frameskip.load(std::memory_order_relaxed) == static_cast<int>(Frameskip::Auto))
                s.autoLevel.store(autoFrameskipStep(s.autoState, vps, target), std::memory_order_relaxed);
            const uint64_t dropped = ps2x::pacing::g_frameskipDropped.exchange(0, std::memory_order_relaxed);
            const uint64_t submitted = ps2x::pacing::g_frameskipSubmitted.exchange(0, std::memory_order_relaxed);
            std::cout << "[savestate] turbo " << static_cast<int>(vps + 0.5) << " VPS, frameskip " << effectiveFrameskip()
                      << ", draws dropped " << dropped << " of " << (dropped + submitted)
                      << ", swaps " << ps2x::pacing::g_frameskipSwaps.exchange(0, std::memory_order_relaxed)
                      << ", dropping frames " << ps2x::pacing::g_frameskipDropFrames.exchange(0, std::memory_order_relaxed)
                      << ", reads back " << ps2x::pacing::g_frameskipReadsBack.load(std::memory_order_relaxed) << std::endl;
            s.vpsSince = now;
            s.vpsTicks = ticks;
        }

        // Messages go to the window title for a few seconds.
        std::lock_guard<std::mutex> lock(s.messageMutex);
        if (s.messageNew)
        {
            s.messageNew = false;
            s.messageUntil = now + kMessageTime;
#if !PS2X_HOST_RAYLIB
            ps2host::setWindowTitleSuffix(s.message.c_str());
#endif
        }
        else if (s.messageUntil != std::chrono::steady_clock::time_point{} && now >= s.messageUntil)
        {
            s.messageUntil = {};
#if !PS2X_HOST_RAYLIB
            ps2host::setWindowTitleSuffix(nullptr);
#endif
        }
    }

    bool skipPresent()
    {
        const int level = effectiveFrameskip();
        return (level == 1 || level == 3) && (svc().hostFrame & 1u) != 0u;
    }
}
