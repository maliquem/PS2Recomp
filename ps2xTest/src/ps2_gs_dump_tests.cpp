// GS dump reader (gs_dump_replay.cpp) on a small synthetic dump written in the PCSX2 layout (GSDump.cpp,
// GSState::Freeze, state version 9). The scene only shows up right if the state, the privileged registers and
// GIF records split in the middle of a tag all reach the GS.

#include "MiniTest.h"
#include "gs_dump_replay.h"

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_types.h"
#include "runtime/gs/gs_vulkan_backend.h"
#include "runtime/gs/ps2_gs_psmct32.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kVram = 4u * 1024u * 1024u;
    constexpr uint32_t kGray = 0x00404040u, kRed = 0x800000FFu, kGreen = 0x8000FF00u;

    struct Bytes
    {
        std::vector<uint8_t> b;
        void raw(const void *p, size_t n) { b.insert(b.end(), static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(p) + n); }
        void u8(uint8_t v) { raw(&v, 1); }
        void u32(uint32_t v) { raw(&v, 4); }
        void u64(uint64_t v) { raw(&v, 8); }
        void qword(uint64_t lo, uint64_t hi) { u64(lo); u64(hi); }
    };

    uint64_t gifTag(uint32_t nloop, uint32_t flg, uint32_t nreg, bool eop)
    {
        return nloop | (eop ? (1ull << 15) : 0ull) | (static_cast<uint64_t>(flg) << 58) | (static_cast<uint64_t>(nreg) << 60);
    }

    void transfer(Bytes &out, uint8_t path, const std::vector<uint8_t> &data, size_t from, size_t to)
    {
        out.u8(0);
        out.u8(path);
        out.u32(static_cast<uint32_t>(to - from));
        out.raw(data.data() + from, to - from);
    }

    // Frame 0 (640x448 CT32) is gray in the saved VRAM, FRAME/SCISSOR/TEST of context 1 exist only in the saved
    // state, the display is turned on by a Registers packet, and two GIF packets are each cut in two records:
    // PATH3 uploads an 8x8 red block at (100,100), PATH2 draws a green sprite (200,50)-(240,90).
    // Record path index as PCSX2 writes it (GSState::Transfer<index>): 1 = PATH2, 2 = PATH3.
    // inFlight: PATH2 is saved in the middle of a PACKED tag (NLOOP 2, NREG 2, reg 1), so its first record starts
    // with 3 leftover qwords that must be skipped, not read as a tag.
    // fieldMode: SMODE2 INT=1 FFMD=1 with the hardware DISPLAY height of the whole frame (DH 447), as Shadow Hearts.
    std::vector<uint8_t> makeSyntheticDump(bool inFlight = false, bool fieldMode = false)
    {
        Bytes state;
        state.u32(9); // STATE_VERSION
        const uint64_t env[15] = {0}; // PRIM .. TRXREG, obsolete TRXREG: all zero
        for (uint64_t v : env)
            state.u64(v);
        const uint64_t frame = 0ull | (10ull << 16) | (static_cast<uint64_t>(GS_PSM_CT32) << 24);
        const uint64_t scissor = (639ull << 16) | (447ull << 48);
        const uint64_t test = 0x30000ull;   // ZTE=1, ZTST=ALWAYS
        const uint64_t zbuf = 1ull << 32;   // ZMSK
        for (int c = 0; c < 2; ++c)
        {
            // XYOFFSET TEX0 TEX1 CLAMP MIPTBP1 MIPTBP2 SCISSOR ALPHA TEST FBA FRAME ZBUF
            const uint64_t ctx[12] = {0, 0, 0, 0, 0, 0, scissor, 0, test, 0, frame, zbuf};
            for (uint64_t v : ctx)
                state.u64(c == 0 ? v : 0ull);
        }
        state.u64(0x3F80000000000000ull | 0x80808080ull); // RGBAQ, Q = 1.0
        state.u64(0);                                      // ST
        state.u32(0);                                      // UV
        state.u32(0);                                      // FOG
        state.u64(0);                                      // XYZ
        state.u64(0);                                      // obsolete
        state.u32(0);                                      // m_tr.x
        state.u32(0);                                      // m_tr.y
        for (int i = 0; i < 61; ++i)                       // m_tr (version 9): idle
            state.u8(0);
        std::vector<uint8_t> vram(kVram, 0u);
        for (uint32_t y = 0; y < 448u; ++y)
            for (uint32_t x = 0; x < 640u; ++x)
                std::memcpy(vram.data() + GSPSMCT32::addrPSMCT32(0u, 10u, x, y), &kGray, 4);
        state.raw(vram.data(), vram.size());
        for (int p = 0; p < 4; ++p) // GIF paths: tag (16) + reg (4); idle unless inFlight
        {
            const bool busy = inFlight && p == 1;
            state.qword(busy ? gifTag(2, GIF_FMT_PACKED, 2, true) : 0ull, busy ? 0xEEull : 0ull);
            state.u32(busy ? 1u : 0u);
        }
        const float q = 1.0f;
        state.raw(&q, 4);

        Bytes out;
        out.u32(0xFFFFFFFFu);
        const std::string serial = "SYNTH-00000";
        out.u32(36u + static_cast<uint32_t>(serial.size()));
        const uint32_t header[9] = {9u, static_cast<uint32_t>(state.b.size()), 36u, static_cast<uint32_t>(serial.size()), 0u, 0u, 0u, 0u, 0u};
        out.raw(header, sizeof(header));
        out.raw(serial.data(), serial.size());
        out.raw(state.b.data(), state.b.size());
        std::vector<uint8_t> priv(0x2000u, 0u); // display off until the Registers packet
        out.raw(priv.data(), priv.size());

        Bytes upload; // PATH3: A+D transfer setup, then IMAGE
        upload.qword(gifTag(4, GIF_FMT_PACKED, 1, false), 0xE);
        upload.qword((0ull << 32) | (10ull << 48) | (static_cast<uint64_t>(GS_PSM_CT32) << 56), GS_REG_BITBLTBUF);
        upload.qword((100ull << 32) | (100ull << 48), GS_REG_TRXPOS);
        upload.qword(8ull | (8ull << 32), GS_REG_TRXREG);
        upload.qword(0, GS_REG_TRXDIR);
        upload.qword(gifTag(16, GIF_FMT_IMAGE, 0, true), 0);
        for (int i = 0; i < 64; ++i)
            upload.u32(kRed);

        Bytes sprite; // PATH2: A+D sprite
        sprite.qword(gifTag(4, GIF_FMT_PACKED, 1, true), 0xE);
        sprite.qword(GS_PRIM_SPRITE, GS_REG_PRIM);
        sprite.qword(kGreen, GS_REG_RGBAQ);
        sprite.qword((200ull << 4) | ((50ull << 4) << 16), GS_REG_XYZ2);
        sprite.qword((240ull << 4) | ((90ull << 4) << 16), GS_REG_XYZ2);

        if (inFlight)
        {
            Bytes leftover; // rest of the saved PACKED tag: A+D writes that would wreck the frame if read as a tag
            for (int i = 0; i < 3; ++i)
                leftover.qword(0x1FFull, GS_REG_FRAME_1);
            transfer(out, 1, leftover.b, 0, leftover.b.size());
        }
        transfer(out, 2, upload.b, 0, 176);      // ends after 5 of the 16 image qwords
        transfer(out, 1, sprite.b, 0, 48);       // ends inside the A+D list
        transfer(out, 2, upload.b, 176, upload.b.size());
        transfer(out, 1, sprite.b, 48, sprite.b.size());

        out.u8(3);
        priv[0x000] = 1u;                                                  // PMODE: EN1
        priv[0x020] = fieldMode ? 3u : 0u;                                 // SMODE2: INT, FFMD
        const uint64_t dispfb = 0ull | (10ull << 9) | (static_cast<uint64_t>(GS_PSM_CT32) << 15);
        const uint64_t display = (639ull << 32) | (447ull << 44);
        std::memcpy(priv.data() + 0x070, &dispfb, 8);
        std::memcpy(priv.data() + 0x080, &display, 8);
        out.raw(priv.data(), priv.size());
        out.u8(1); out.u8(0); // VSync, field 0
        out.u8(3);
        out.raw(priv.data(), priv.size());
        out.u8(1); out.u8(1); // VSync, field 1
        return out.b;
    }

    uint32_t rgbAt(const gsdump::Frame &f, uint32_t x, uint32_t y)
    {
        uint32_t v = 0;
        std::memcpy(&v, f.rgba.data() + (static_cast<size_t>(y) * f.width + x) * 4u, 4);
        return v & 0x00FFFFFFu;
    }

    void checkScene(TestCase &t, const gsdump::Result &r, const char *backend, bool inFlight = false,
                    bool fieldMode = false)
    {
        const std::string tag = std::string(" (") + backend + ")";
        t.IsTrue(r.ok, ("replay should succeed: " + r.error + tag).c_str());
        t.Equals(r.warnings.size(), static_cast<size_t>(inFlight ? 1 : 0), ("warning only for the packet in flight" + tag).c_str());
        t.Equals(r.serial, std::string("SYNTH-00000"), ("serial from the header" + tag).c_str());
        t.Equals(r.transfers, inFlight ? 5u : 4u, ("transfer records" + tag).c_str());
        t.Equals(r.frames.size(), static_cast<size_t>(2), ("one frame per VSync" + tag).c_str());
        for (const gsdump::Frame &f : r.frames)
        {
            t.Equals(f.width, 640u, ("frame width from DISPLAY1" + tag).c_str());
            const uint32_t height = fieldMode ? 224u : 448u; // field mode shows one field: half of DH+1
            t.Equals(f.height, height, ("frame height from DISPLAY1" + tag).c_str());
            t.Equals(f.field, fieldMode, ("field flag for the comparator" + tag).c_str());
            if (f.width != 640u || f.height != height)
                return;
            t.Equals(rgbAt(f, 10, 10), kGray & 0xFFFFFFu, ("background comes from the saved VRAM" + tag).c_str());
            t.Equals(rgbAt(f, 104, 104), kRed & 0xFFFFFFu, ("PATH3 upload split across records lands" + tag).c_str());
            t.Equals(rgbAt(f, 107, 107), kRed & 0xFFFFFFu, ("last uploaded pixel lands" + tag).c_str());
            t.Equals(rgbAt(f, 109, 104), kGray & 0xFFFFFFu, ("upload stays 8 px wide" + tag).c_str());
            t.Equals(rgbAt(f, 220, 70), kGreen & 0xFFFFFFu, ("PATH2 sprite drawn with the saved FRAME" + tag).c_str());
            t.Equals(rgbAt(f, 245, 95), kGray & 0xFFFFFFu, ("sprite stays in its rectangle" + tag).c_str());
        }
    }
}

void register_ps2_gs_dump_tests()
{
    MiniTest::Case("PS2GSDump", [](TestCase &tc)
    {
        tc.Run("synthetic PCSX2 GS dump replays to the expected frame on the CPU backend", [](TestCase &t)
        {
            const std::vector<uint8_t> dump = makeSyntheticDump();
            // Through a file, as ps2x_gsdump reads it; kept in the temp folder for a manual ps2x_gsdump run.
            const std::filesystem::path path = std::filesystem::temp_directory_path() / "ps2x_gsdump_synthetic.gs";
            std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(dump.data()),
                                                         static_cast<std::streamsize>(dump.size()));
            std::vector<uint8_t> fromFile;
            t.IsTrue(gsdump::readFile(path, fromFile), "synthetic dump file reads back");
            t.IsTrue(fromFile == dump, "file holds the synthetic dump");
            checkScene(t, gsdump::replay(fromFile, std::make_unique<GSCpuBackend>()), "cpu");
        });

        tc.Run("synthetic PCSX2 GS dump replays to the expected frame on the Vulkan backend", [](TestCase &t)
        {
            auto vk = GSVulkanBackend::CreateIfAvailable();
            if (!vk)
                return; // headless machine without a Vulkan compute device, as the other Vulkan tests
            checkScene(t, gsdump::replay(makeSyntheticDump(), std::move(vk)), "vulkan");
        });

        tc.Run("GS dump reader skips the rest of a packet in flight at dump start", [](TestCase &t)
        {
            checkScene(t, gsdump::replay(makeSyntheticDump(true), std::make_unique<GSCpuBackend>()), "cpu", true);
        });

        tc.Run("GS dump in field mode presents one field of the frame", [](TestCase &t)
        {
            checkScene(t, gsdump::replay(makeSyntheticDump(false, true), std::make_unique<GSCpuBackend>()), "cpu",
                       false, true);
        });

        tc.Run("GS dump reader rejects compressed and truncated dumps", [](TestCase &t)
        {
            const std::vector<uint8_t> zstd = {0x28, 0xB5, 0x2F, 0xFD, 0, 0, 0, 0};
            const gsdump::Result compressed = gsdump::replay(zstd, nullptr);
            t.IsFalse(compressed.ok, "zstd dump is refused");
            t.IsTrue(compressed.error.find("Uncompressed") != std::string::npos, "error tells how to record it");

            std::vector<uint8_t> cut = makeSyntheticDump();
            cut.resize(4096);
            const gsdump::Result truncated = gsdump::replay(cut, nullptr);
            t.IsFalse(truncated.ok, "dump cut inside the state is refused");

            Bytes shortHeader; // header says 47 bytes with an 11-byte serial, the file stops after 42
            shortHeader.u32(0xFFFFFFFFu);
            shortHeader.u32(47u);
            const uint32_t header[9] = {9u, 0u, 36u, 11u, 0u, 0u, 0u, 0u, 0u};
            shortHeader.raw(header, sizeof(header));
            shortHeader.raw("SYNTH-", 6);
            const gsdump::Result cutHeader = gsdump::replay(shortHeader.b, nullptr);
            t.IsFalse(cutHeader.ok, "dump cut inside the header is refused");
            t.IsTrue(cutHeader.serial.empty(), "serial is not read past the end of the file");
        });
    });
}
