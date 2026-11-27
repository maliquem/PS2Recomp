#include "gs_dump_replay.h"

#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace gsdump
{
    namespace
    {
        constexpr uint32_t kVramBytes = 4u * 1024u * 1024u;
        constexpr uint32_t kPrivRegBytes = 0x2000u; // GSPrivRegSet
        constexpr uint32_t kDumpHeaderBytes = 36u;  // GSDumpHeader: 9 x u32
        constexpr uint32_t kTransferStateV9Bytes = 61u; // m_tr.w .. m_tr.write (GSState::Freeze, version 9)

        struct Cursor
        {
            const uint8_t *data;
            size_t pos;
            size_t end;

            bool take(void *out, size_t n)
            {
                if (n > end - pos)
                    return false;
                std::memcpy(out, data + pos, n);
                pos += n;
                return true;
            }
            template <class T> bool get(T &v) { return take(&v, sizeof(T)); }
            bool skip(size_t n)
            {
                if (n > end - pos)
                    return false;
                pos += n;
                return true;
            }
        };

        uint64_t le64(const uint8_t *p)
        {
            uint64_t v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }

        // GSPrivRegSet keeps each register at its hardware offset from 0x12000000.
        void loadPrivRegs(GSRegisters &r, const uint8_t *p)
        {
            r.pmode = le64(p + 0x000);
            r.smode1 = le64(p + 0x010);
            r.smode2 = le64(p + 0x020);
            r.srfsh = le64(p + 0x030);
            r.synch1 = le64(p + 0x040);
            r.synch2 = le64(p + 0x050);
            r.syncv = le64(p + 0x060);
            r.dispfb1 = le64(p + 0x070);
            r.display1 = le64(p + 0x080);
            r.dispfb2 = le64(p + 0x090);
            r.display2 = le64(p + 0x0A0);
            r.extbuf = le64(p + 0x0B0);
            r.extdata = le64(p + 0x0C0);
            r.extwrite = le64(p + 0x0D0);
            r.bgcolor = le64(p + 0x0E0);
            r.csr.store(le64(p + 0x1000));
            r.imr = le64(p + 0x1010);
            r.busdir = le64(p + 0x1040);
            r.siglblid = le64(p + 0x1080);
        }

        // Bytes of one GIF tag and its data; same size rules as GS::processGIFPacket.
        uint64_t tagUnitBytes(uint64_t tagLo)
        {
            const uint64_t nloop = tagLo & 0x7FFFu;
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3u);
            uint64_t nreg = (tagLo >> 60) & 0xFu;
            if (nreg == 0u)
                nreg = 16u;
            if (flg == GIF_FMT_PACKED)
                return 16u + nloop * nreg * 16u;
            if (flg == GIF_FMT_REGLIST)
                return 16u + ((nloop * nreg + 1u) / 2u) * 16u;
            return 16u + nloop * 16u; // IMAGE and DISABLED
        }

        // PCSX2 dumps each GIF transfer as it arrived, so a record can end inside a tag and the next record of the
        // same path continues it. GS::processGIFPacket starts at a tag, so only whole tags are passed on.
        // shortcut: a tag split across records waits for its rest, so records of other paths in between run before
        // its first half (PCSX2 ran them after). Harmless for whole packets; revisit if a sliced PATH3 IMAGE that
        // another path draws from shows up in a dump.
        struct PathFeed
        {
            std::vector<uint8_t> pending;
            uint64_t skipBytes = 0; // rest of a packet that was in flight when the dump started
        };

        void feedPath(GS &gs, PathFeed &path, const uint8_t *data, size_t size)
        {
            const size_t skipped = static_cast<size_t>(std::min<uint64_t>(path.skipBytes, size));
            path.skipBytes -= skipped;
            path.pending.insert(path.pending.end(), data + skipped, data + size);

            size_t complete = 0;
            while (complete + 16u <= path.pending.size())
            {
                const uint64_t unit = tagUnitBytes(le64(path.pending.data() + complete));
                if (unit > path.pending.size() - complete)
                    break;
                complete += static_cast<size_t>(unit);
            }
            if (complete == 0u)
                return;
            gs.processGIFPacket(path.pending.data(), static_cast<uint32_t>(complete));
            path.pending.erase(path.pending.begin(), path.pending.begin() + static_cast<std::ptrdiff_t>(complete));
        }

        struct EnvState
        {
            uint64_t prim, prmodecont, texclut, scanmsk, texa, fogcol, dimx, dthe, colclamp, pabe;
            uint64_t bitbltbuf, trxdir, trxpos, trxreg;
            struct Ctx
            {
                uint64_t xyoffset, tex0, tex1, clamp, miptbp1, miptbp2, scissor, alpha, test, fba, frame, zbuf;
            } ctx[2];
            uint64_t rgbaq, st;
            uint32_t uv, fog;
        };

        // Restores the registers the way GSState::Defrost does, through the same writes a game makes. TRXDIR is not
        // written (it would start a transfer) and XYZ is not written (it would kick a vertex).
        void applyEnv(GS &gs, const EnvState &e)
        {
            gs.writeRegister(GS_REG_PRMODECONT, e.prmodecont);
            gs.writeRegister(GS_REG_PRMODE, e.prim); // with AC=0 PCSX2 keeps the PRMODE attributes inside PRIM
            gs.writeRegister(GS_REG_PRIM, e.prim);
            gs.writeRegister(GS_REG_TEXCLUT, e.texclut);
            gs.writeRegister(GS_REG_SCANMSK, e.scanmsk);
            gs.writeRegister(GS_REG_TEXA, e.texa);
            gs.writeRegister(GS_REG_FOGCOL, e.fogcol);
            gs.writeRegister(GS_REG_DIMX, e.dimx);
            gs.writeRegister(GS_REG_DTHE, e.dthe);
            gs.writeRegister(GS_REG_COLCLAMP, e.colclamp);
            gs.writeRegister(GS_REG_PABE, e.pabe);
            gs.writeRegister(GS_REG_BITBLTBUF, e.bitbltbuf);
            gs.writeRegister(GS_REG_TRXPOS, e.trxpos);
            gs.writeRegister(GS_REG_TRXREG, e.trxreg);
            for (int i = 0; i < 2; ++i)
            {
                const EnvState::Ctx &c = e.ctx[i];
                const auto w = [&](int reg, uint64_t v) { gs.writeRegister(static_cast<uint8_t>(reg + i), v); };
                w(GS_REG_XYOFFSET_1, c.xyoffset);
                w(GS_REG_TEX1_1, c.tex1);
                w(GS_REG_CLAMP_1, c.clamp);
                w(GS_REG_MIPTBP1_1, c.miptbp1);
                w(GS_REG_MIPTBP2_1, c.miptbp2);
                w(GS_REG_SCISSOR_1, c.scissor);
                w(GS_REG_ALPHA_1, c.alpha);
                w(GS_REG_TEST_1, c.test);
                w(GS_REG_FBA_1, c.fba);
                w(GS_REG_FRAME_1, c.frame);
                w(GS_REG_ZBUF_1, c.zbuf);
            }
            gs.writeRegister(GS_REG_RGBAQ, e.rgbaq);
            gs.writeRegister(GS_REG_ST, e.st);
            gs.writeRegister(GS_REG_UV, e.uv);
            gs.writeRegister(GS_REG_FOG, static_cast<uint64_t>(e.fog & 0xFFu) << 56);

            // Defrost reloads the CLUT from the active context's TEX0 only; TEX0 writes load it here, so the other
            // context goes first with CLD=0 (no load) and the active one last with its own CLD.
            const int active = ((e.prim >> 9) & 1u) ? 1 : 0;
            constexpr uint64_t kCldMask = 7ull << 61;
            gs.writeRegister(static_cast<uint8_t>(GS_REG_TEX0_1 + (1 - active)), e.ctx[1 - active].tex0 & ~kCldMask);
            gs.writeRegister(static_cast<uint8_t>(GS_REG_TEX0_1 + active), e.ctx[active].tex0);
        }

        void addWarning(Result &r, const char *fmt, unsigned long long a, unsigned long long b = 0,
                        unsigned long long c = 0, unsigned long long d = 0)
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf), fmt, a, b, c, d);
            r.warnings.emplace_back(buf);
        }
    }

    bool readFile(const std::filesystem::path &path, std::vector<uint8_t> &out)
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in)
            return false;
        const std::streamoff size = in.tellg();
        if (size < 0)
            return false;
        out.resize(static_cast<size_t>(size));
        in.seekg(0);
        return static_cast<bool>(in.read(reinterpret_cast<char *>(out.data()), size));
    }

    Result replay(const std::vector<uint8_t> &file, std::unique_ptr<GSRasterBackend> backend)
    {
        Result r;
        Cursor cur{file.data(), 0u, file.size()};

        uint32_t magic = 0, headerSize = 0;
        if (!cur.get(magic) || magic != 0xFFFFFFFFu)
        {
            r.error = (magic == 0xFD2FB528u || magic == 0x587A37FDu) // zstd, xz
                          ? "compressed GS dump; record it again with GS Dump Compression = Uncompressed"
                          : "not a PCSX2 GS dump of the current format (first word is not 0xFFFFFFFF)";
            return r;
        }
        uint32_t header[9] = {};
        if (!cur.get(headerSize) || headerSize < kDumpHeaderBytes || !cur.take(header, sizeof(header)))
        {
            r.error = "truncated GS dump header";
            return r;
        }
        const uint32_t stateVersion = header[0], stateSize = header[1];
        const uint32_t serialOffset = header[2], serialSize = header[3];
        const size_t headerStart = cur.pos - kDumpHeaderBytes;
        if (serialSize != 0u && serialOffset <= headerSize && serialSize <= headerSize - serialOffset)
            r.serial.assign(reinterpret_cast<const char *>(file.data() + headerStart + serialOffset), serialSize);
        if (!cur.skip(headerSize - kDumpHeaderBytes))
        {
            r.error = "truncated GS dump header";
            return r;
        }

        const size_t stateStart = cur.pos;
        uint32_t version = 0;
        cur.get(version);
        r.stateVersion = version;
        if (version != stateVersion || version < 7u || version > 9u)
        {
            r.error = "unsupported GS state version " + std::to_string(version) + " (this reader knows 7 to 9)";
            return r;
        }

        EnvState env{};
        uint64_t unused64 = 0;
        uint32_t trX = 0, trY = 0;
        bool ok = cur.get(env.prim) && cur.get(env.prmodecont) && cur.get(env.texclut) && cur.get(env.scanmsk) &&
                  cur.get(env.texa) && cur.get(env.fogcol) && cur.get(env.dimx) && cur.get(env.dthe) &&
                  cur.get(env.colclamp) && cur.get(env.pabe) && cur.get(env.bitbltbuf) && cur.get(env.trxdir) &&
                  cur.get(env.trxpos) && cur.get(env.trxreg) && cur.get(unused64);
        for (auto &c : env.ctx)
            ok = ok && cur.get(c.xyoffset) && cur.get(c.tex0) && cur.get(c.tex1) && cur.get(c.clamp) &&
                 cur.get(c.miptbp1) && cur.get(c.miptbp2) && cur.get(c.scissor) && cur.get(c.alpha) &&
                 cur.get(c.test) && cur.get(c.fba) && cur.get(c.frame) && cur.get(c.zbuf);
        ok = ok && cur.get(env.rgbaq) && cur.get(env.st) && cur.get(env.uv) && cur.get(env.fog) &&
             cur.get(unused64) /* XYZ */ && cur.get(unused64) /* obsolete */ && cur.get(trX) && cur.get(trY);

        if (ok && version >= 9u)
        {
            // m_tr: w, h (int), BITBLTBUF, TRXPOS, TRXREG (u64), rect (4 x int), total, start, end (int), write (bool).
            uint8_t tr[kTransferStateV9Bytes];
            ok = cur.take(tr, sizeof(tr));
            int32_t total = 0, end = 0;
            std::memcpy(&total, tr + 48, 4);
            std::memcpy(&end, tr + 56, 4);
            if (ok && tr[60] != 0u && total > 0 && end < total)
                addWarning(r, "a host-to-local transfer was in flight at dump start (%llu of %llu bytes); its rest is lost",
                           static_cast<unsigned long long>(end), static_cast<unsigned long long>(total));
        }

        std::vector<uint8_t> vram(kVramBytes);
        ok = ok && cur.take(vram.data(), kVramBytes);

        std::array<PathFeed, 4> paths{};
        for (uint32_t i = 0; ok && i < 4u; ++i)
        {
            uint64_t tagLo = 0, tagHi = 0;
            uint32_t reg = 0;
            ok = cur.get(tagLo) && cur.get(tagHi) && cur.get(reg);
            const uint64_t nloop = tagLo & 0x7FFFu;
            if (!ok || nloop == 0u)
                continue;
            // Freeze stores the loops left in NLOOP and the register index inside the current loop in reg.
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3u);
            uint64_t nreg = (tagLo >> 60) & 0xFu;
            if (nreg == 0u)
                nreg = 16u;
            const uint64_t regsLeft = nloop * nreg - std::min<uint64_t>(reg, nreg);
            uint64_t rest = nloop * 16u;
            if (flg == GIF_FMT_PACKED)
                rest = regsLeft * 16u;
            else if (flg == GIF_FMT_REGLIST)
                rest = (regsLeft * 8u + 15u) & ~15ull;
            paths[i].skipBytes = rest;
            addWarning(r, "path %llu had a packet in flight at dump start (FLG %llu, NLOOP %llu); its rest (%llu bytes) is skipped",
                       i, flg, nloop, rest);
        }
        float q = 0.0f;
        ok = ok && cur.get(q);
        if (!ok || cur.pos - stateStart != stateSize)
        {
            r.error = "GS state layout mismatch: read " + std::to_string(cur.pos - stateStart) + " bytes, header says " +
                      std::to_string(stateSize);
            return r;
        }

        GSRegisters regs{};
        std::vector<uint8_t> priv(kPrivRegBytes);
        if (!cur.take(priv.data(), kPrivRegBytes))
        {
            r.error = "truncated privileged registers";
            return r;
        }
        loadPrivRegs(regs, priv.data());

        // The backend picks the VRAM up from the buffer when it is attached (Vulkan uploads it then).
        auto gsOwner = std::make_unique<GS>(); // large object (debug history): keep it off the stack
        GS &gs = *gsOwner;
        gs.init(vram.data(), kVramBytes, &regs);
        if (backend)
            gs.setRasterBackend(std::move(backend));
        applyEnv(gs, env);

        std::vector<uint8_t> fifo;
        while (cur.pos < cur.end)
        {
            const size_t packetPos = cur.pos;
            uint8_t id = 0;
            cur.get(id);
            bool truncated = false;
            switch (id)
            {
            case 0: // Transfer: path, size, data
            {
                uint8_t path = 0;
                uint32_t size = 0;
                if (!cur.get(path) || !cur.get(size) || size > cur.end - cur.pos)
                {
                    truncated = true;
                    break;
                }
                feedPath(gs, paths[path & 3u], file.data() + cur.pos, size);
                cur.pos += size;
                ++r.transfers;
                break;
            }
            case 1: // VSync: field
            {
                uint8_t field = 0;
                if (!cur.get(field))
                {
                    truncated = true;
                    break;
                }
                regs.vsyncTick.store(field, std::memory_order_release); // presentation picks the field by parity
                gs.latchHostPresentationFrame();
                Frame f;
                gs.copyLatchedHostPresentationFrame(f.rgba, f.width, f.height, &f.displayFbp, &f.sourceFbp,
                                                    &f.usedPreferred);
                const uint64_t dispfb = (regs.pmode & 1u) ? regs.dispfb1 : regs.dispfb2;
                f.displayPsm = static_cast<uint32_t>((dispfb >> 15) & 0x1Fu);
                r.frames.push_back(std::move(f));
                break;
            }
            case 2: // ReadFIFO2: size in qwords; the local-to-host data is drained as the EE would
            {
                uint32_t qwords = 0;
                if (!cur.get(qwords))
                {
                    truncated = true;
                    break;
                }
                fifo.resize(static_cast<size_t>(qwords) * 16u);
                if (!fifo.empty())
                    gs.consumeLocalToHostBytes(fifo.data(), static_cast<uint32_t>(fifo.size()));
                ++r.readFifos;
                break;
            }
            case 3: // Registers (before each VSync)
                if (!cur.take(priv.data(), kPrivRegBytes))
                {
                    truncated = true;
                    break;
                }
                loadPrivRegs(regs, priv.data());
                break;
            default:
                r.error = "unknown GS dump packet id " + std::to_string(id) + " at offset " + std::to_string(packetPos);
                return r;
            }
            if (truncated)
            {
                addWarning(r, "dump ends inside a packet (id %llu at offset %llu); the rest is ignored", id, packetPos);
                break;
            }
        }

        for (uint32_t i = 0; i < 4u; ++i)
            if (!paths[i].pending.empty())
                addWarning(r, "path %llu ended with %llu bytes of an incomplete GIF packet", i, paths[i].pending.size());
        r.ok = true;
        return r;
    }
}
