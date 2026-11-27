#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <ps2_log.h>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    m_flagPipeline.clear();
    m_fdiv = {};
    m_efu = {};
    m_storePipeline.clear();
    m_vfWritePipeline.clear();
    m_viWritePipeline.clear();
    m_xgkick.reset();
    m_vfReady = {};
    m_viReady = {};
    m_vfLatestWrite = {};
    m_viLatestWrite = {};
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint32_t op = upper & 0x3Fu;
    const uint32_t special = (upper & 3u) | ((upper >> 4) & 0x7Cu);
    const uint32_t arithmetic = op >= 0x3Cu ? special : op;
    const bool simple = arithmetic <= 0x07u ||
                        (arithmetic >= 0x18u && arithmetic <= 0x1Cu) ||
                        arithmetic == 0x1Eu || arithmetic == 0x20u ||
                        arithmetic == 0x22u || arithmetic == 0x24u ||
                        arithmetic == 0x26u || arithmetic == 0x28u ||
                        arithmetic == 0x2Au || arithmetic == 0x2Cu ||
                        (op >= 0x3Cu && arithmetic == 0x2Eu);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        if (simple)
        {
            const uint32_t bits = std::bit_cast<uint32_t>(result[component]);
            const uint32_t magnitude = bits & 0x7FFFFFFFu;
            // A single operation rounded strictly inside the normal range can only set S.
            if (magnitude > 0x00800000u && magnitude < 0x7F7FFFFFu)
            {
                laneFlags[component] = static_cast<uint8_t>((bits >> 30) & 2u);
                continue;
            }
        }

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    return vuNormalizeFmacExactResult(value, exactResult);
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = m_flagPipeline.schedule();
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    m_flagPipeline.forEach([&](FlagPipelineEntry &entry)
    {
        if (entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    });

    if (auto *slot = m_flagPipeline.schedule())
    {
        auto &entry = *slot;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry.writesSticky = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (auto *slot = m_flagPipeline.schedule())
    {
        auto &entry = *slot;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        entry.clip = m_workingClip;
        entry.writesClip = true;
        return;
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    m_flagPipeline.forEach([&](FlagPipelineEntry &entry)
    {
        if (entry.issueCycle == m_cycle)
            entry.writesClip = false;
    });
    if (auto *slot = m_flagPipeline.schedule())
    {
        auto &entry = *slot;
        entry.issueCycle = m_cycle;
        entry.readyCycle = m_cycle + kFmacLatency;
        entry.clip = m_workingClip;
        entry.writesClip = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (auto *slot = m_storePipeline.schedule())
    {
        auto &store = *slot;
        store.readyCycle = m_cycle + 1u;
        store.address = address;
        store.laneMask = laneMask;
        std::copy(words, words + 4, store.words.begin());
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    if (auto *slot = m_vfWritePipeline.schedule())
    {
        auto &write = *slot;
        write.readyCycle = m_cycle + latency;
        write.sequence = ++m_nextWriteSequence;
        write.reg = reg;
        write.laneMask = laneMask;
        std::copy(value, value + 4, write.value.begin());
        auto &ready = m_vfReady[reg];
        auto &latest = m_vfLatestWrite[reg];
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((laneMask & laneForComponent(component)) != 0u)
            {
                ready[component] = write.readyCycle;
                latest[component] = write.sequence;
            }
        }
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    if (auto *slot = m_viWritePipeline.schedule())
    {
        auto &write = *slot;
        write.readyCycle = m_cycle + latency;
        write.sequence = ++m_nextWriteSequence;
        write.reg = reg;
        write.value = value;
        m_viReady[reg] = write.readyCycle;
        m_viLatestWrite[reg] = write.sequence;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::commitReadyPipelines()
{
    m_flagPipeline.commit(m_cycle, [&](const FlagPipelineEntry &entry)
    {

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
    });

    if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
    {
        m_state.q = m_fdiv.value;
        const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
        m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
        m_fdiv = {};
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (entry.valid && entry.readyCycle <= m_cycle)
        {
            m_state.p = entry.value;
            entry = {};
        }
    }

    m_storePipeline.commit(m_cycle, [&](const PendingStore &store)
    {
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            uint32_t oldWords[4]{};
            std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((store.laneMask & laneForComponent(component)) != 0u)
                    oldWords[component] = store.words[component];
            }
            std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
        }
    });

    m_vfWritePipeline.commit(m_cycle, [&](const PendingVfWrite &write)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
    });

    m_viWritePipeline.commit(m_cycle, [&](const PendingViWrite &write)
    {
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
    });
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t source = (m_xgkick.sourceAddress + qwordOffset) % m_activeVuDataSize;
        const uint32_t firstSize = std::min(16u, m_activeVuDataSize - source);
        uint8_t *destination = m_xgkick.packet.data() + qwordOffset;
        std::memcpy(destination, m_activeVuData + source, firstSize);
        if (firstSize != 16u)
            std::memcpy(destination + firstSize, m_activeVuData, 16u - firstSize);
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u || format == 3u) // FLG=3 ("disabled" in the GS manual) behaves exactly like IMAGE: NLOOP qwords of image data.
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    m_xgkick.reset();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    return m_flagPipeline.pending() || m_storePipeline.pending() ||
           m_vfWritePipeline.pending() || m_viWritePipeline.pending();
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending())
        advanceOneCycle();
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const InstructionUsage *usages[2] = {
        &decoded.upperUsage,
        &decoded.lowerUsage};
    for (const InstructionUsage *usage : usages)
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((access.lanes & laneForComponent(component)) != 0u)
                    ready = std::max(ready, m_vfReady[access.reg][component]);
            }
        }
        for (uint32_t pendingReads = usage->viRead & ~1u; pendingReads != 0u; pendingReads &= pendingReads - 1u)
        {
            ready = std::max(ready, m_viReady[std::countr_zero(pendingReads)]);
        }
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[lowerWrite.reg][component] = m_cycle + latency;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[upperWrite.reg][component] = m_cycle + latency;
        }
    }

    for (uint32_t pendingWrites = decoded.lowerUsage.viWrite & ~1u;
         pendingWrites != 0u; pendingWrites &= pendingWrites - 1u)
    {
        m_viReady[std::countr_zero(pendingWrites)] = m_cycle +
            (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
    }
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special == 0x2Fu || special == 0x30u)
        {
            usage.pipeline = PipelineNone;
        }
        else
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    for (uint32_t i = 0; i < pairCount; ++i)
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);

    m_fastEntryStatus.fill(0u);
    m_fastNodes.clear();
    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

const VU1Interpreter::DecodedInstructionPair *VU1Interpreter::getDecodedCodeCache(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory)
{
    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return nullptr;

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    return m_decodedCodeCache.data();
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    const auto *decodedCode = getDecodedCodeCache(vuCode, codeSize, memory);
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);

    if (m_unit == Unit::VU1 &&
        decodedCode != nullptr &&
        vuData != nullptr &&
        dataSize >= 16u &&
        (dataSize & (dataSize - 1u)) == 0u &&
        maxCycles >= 16384u &&
        !m_stopRequested &&
        !m_state.dBitEnabled &&
        !m_state.tBitEnabled &&
        !m_state.branchPending &&
        !m_state.ebit &&
        !m_state.haltAfterDelaySlot &&
        !pipelinesPending() &&
        (m_state.pc & 7u) == 0u &&
        (m_state.pc / 8u) < pairCount &&
        // The old-VI copy left by a delay-slot write only matters to a first pair that reads that VI.
        (!m_viBranchBackupValid ||
         (decodedCode[m_state.pc / 8u].lowerUsage.viRead & (1u << m_viBranchBackupReg)) == 0u) &&
        isFastEntryEligible(m_state.pc / 8u, pairCount))
    {
        m_viBranchBackupValid = false;
        runFastDirect(vuData, dataSize, pairCount, budgetEnd);
        if (useVuRounding && previousRoundingMode != -1)
            std::fesetround(previousRoundingMode);
        return;
    }

    bool programEnded = false;
    // advanceOneCycle commits pending writes before the next pair issues.
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        if (m_state.pc + 8u > codeSize)
            break;

        const uint32_t pc = m_state.pc;
        DecodedInstructionPair uncached;
        const DecodedInstructionPair *pair;
        if (decodedCode && (pc & 7u) == 0u && pc / 8u < kMaxDecodedPairs)
            pair = &decodedCode[pc / 8u];
        else
        {
            uncached = decodeInstructionPair(vuCode, pc);
            pair = &uncached;
        }
        const DecodedInstructionPair &decoded = *pair;
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = calculatePairReadyCycle(decoded);
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
        }
        if (m_cycle >= budgetEnd)
            break;

        const uint32_t viWriteMask = decoded.lowerUsage.viWrite & ~1u;
        const uint8_t writtenVi = viWriteMask != 0u ? static_cast<uint8_t>(std::countr_zero(viWriteMask)) : 0u;
        const int32_t oldVi = m_state.vi[writtenVi];

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = upperWrite.reg != 0u;
        const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        float oldUpperVf[4]{};
        float newUpperVf[4]{};
        float oldLowerVf[4]{};
        float newLowerVf[4]{};
        if (hasUpperWrite)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (hasDistinctLowerWrite)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));

        // The lower instruction never reads ACC, and each pair advances a cycle before the next upper reads it.
        if (decoded.iBit)
        {
            execUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
            if (decoded.upperUsage.pipeline == PipelineNone)
                m_currentUpperInstruction = decoded.upper;
            else
                execUpper(decoded.upper);
            if (decoded.lowerUsage.pipeline != PipelineNone)
                execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        m_viBranchBackupValid = false;

        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (writtenVi != 0u)
        {
            const int32_t newVi = m_state.vi[writtenVi];
            m_state.vi[writtenVi] = oldVi;
            const uint32_t latency =
                decoded.lowerUsage.viLatency != 0u
                    ? decoded.lowerUsage.viLatency
                    : decoded.lowerUsage.latency;
            queueViWrite(writtenVi, newVi, latency);
        }

        // Preserve the scoreboard on a stopped pair whose queue write could not be scheduled.
        if (m_stopRequested)
            markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        advanceOneCycle();
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}

namespace
{
    enum FastUpperOp : uint8_t
    {
        FU_NOP = 0,
        FU_ADDbc,
        FU_SUBbc,
        FU_MADDbc,
        FU_MSUBbc,
        FU_MAXbc,
        FU_MINIbc,
        FU_MULbc,
        FU_MULq,
        FU_MAXi,
        FU_MULi,
        FU_MINIi,
        FU_ADDq,
        FU_MADDq,
        FU_ADDi,
        FU_MADDi,
        FU_SUBq,
        FU_MSUBq,
        FU_SUBi,
        FU_MSUBi,
        FU_ADD,
        FU_MADD,
        FU_MUL,
        FU_MAX,
        FU_SUB,
        FU_MSUB,
        FU_OPMSUB,
        FU_MINI,
        FU_ADDAbc,
        FU_SUBAbc,
        FU_MADDAbc,
        FU_MSUBAbc,
        FU_MULAbc,
        FU_MULAq,
        FU_MULAi,
        FU_ADDAq,
        FU_MADDAq,
        FU_ADDAi,
        FU_MADDAi,
        FU_SUBAq,
        FU_MSUBAq,
        FU_SUBAi,
        FU_MSUBAi,
        FU_ADDA,
        FU_MADDA,
        FU_MULA,
        FU_SUBA,
        FU_MSUBA,
        FU_OPMULA,
        FU_ITOF0,
        FU_ITOF4,
        FU_ITOF12,
        FU_ITOF15,
        FU_FTOI0,
        FU_FTOI4,
        FU_FTOI12,
        FU_FTOI15,
        FU_ABS,
        FU_CLIPw
    };

    enum FastLowerOp : uint8_t
    {
        FL_NOP = 0,
        FL_LOI,
        FL_LQ,
        FL_SQ,
        FL_ILW,
        FL_ISW,
        FL_LQI,
        FL_SQI,
        FL_LQD,
        FL_SQD,
        FL_ILWR,
        FL_ISWR,
        FL_MOVE,
        FL_MR32,
        FL_MTIR,
        FL_MFIR,
        FL_IADD,
        FL_ISUB,
        FL_IADDI,
        FL_IAND,
        FL_IOR,
        FL_IADDIU,
        FL_ISUBIU,
        FL_DIV,
        FL_SQRT,
        FL_RSQRT,
        FL_WAITQ,
        FL_FCEQ,
        FL_FCAND,
        FL_FCOR,
        FL_B,
        FL_IBEQ,
        FL_IBNE,
        FL_IBLTZ,
        FL_IBGTZ,
        FL_IBLEZ,
        FL_IBGEZ,
        FL_XTOP,
        FL_XITOP,
        FL_XGKICK,
        FL_FCSET,
        FL_FMEQ,
        FL_FMAND,
        FL_FMOR
    };

    constexpr uint8_t kFastFlagEBit = 1u << 0;
    constexpr uint8_t kFastFlagShadowLowerFirst = 1u << 1;
    constexpr uint8_t kFastFlagIsBranch = 1u << 2;
    constexpr uint8_t kFastFlagUncondBranch = 1u << 3;
    constexpr uint8_t kFastFlagReadsQ = 1u << 4;
    constexpr uint8_t kFastFlagIsStore = 1u << 5;

    inline int32_t fastVuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

#if defined(__SSE2__) || defined(_M_X64)
    alignas(16) constexpr uint32_t kFastLaneMasks[16][4] = {
        {0u, 0u, 0u, 0u},
        {0u, 0u, 0u, ~0u},
        {0u, 0u, ~0u, 0u},
        {0u, 0u, ~0u, ~0u},
        {0u, ~0u, 0u, 0u},
        {0u, ~0u, 0u, ~0u},
        {0u, ~0u, ~0u, 0u},
        {0u, ~0u, ~0u, ~0u},
        {~0u, 0u, 0u, 0u},
        {~0u, 0u, 0u, ~0u},
        {~0u, 0u, ~0u, 0u},
        {~0u, 0u, ~0u, ~0u},
        {~0u, ~0u, 0u, 0u},
        {~0u, ~0u, 0u, ~0u},
        {~0u, ~0u, ~0u, 0u},
        {~0u, ~0u, ~0u, ~0u},
    };

    struct ScopedVuMxcsr
    {
        unsigned int saved;
        ScopedVuMxcsr() : saved(_mm_getcsr())
        {
            // Flush-to-zero (bit 15), Round-toward-zero (bits 14:13), Denormals-are-zero (bit 6).
            _mm_setcsr((saved & ~0x6000u) | 0xE040u);
        }
        ~ScopedVuMxcsr()
        {
            _mm_setcsr(saved);
        }
    };

    inline __m128 sseBroadcastLane(__m128 v, uint8_t lane)
    {
        switch (lane)
        {
        case 0: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        case 1: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        case 2: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
        default: return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
        }
    }

    inline void sseApplyMask(__m128 &dst, __m128 val, uint8_t dest)
    {
        if (dest == 0xFu)
        {
            dst = val;
            return;
        }
        const __m128 mask = _mm_load_ps(reinterpret_cast<const float *>(kFastLaneMasks[dest]));
        dst = _mm_or_ps(_mm_and_ps(mask, val), _mm_andnot_ps(mask, dst));
    }

    inline __m128 sseClampVuInf(__m128 v)
    {
        const __m128i bits = _mm_castps_si128(v);
        const __m128i expMask = _mm_set1_epi32(0x7F800000);
        const __m128i isInfOrNan = _mm_cmpeq_epi32(_mm_and_si128(bits, expMask), expMask);
        if (_mm_movemask_epi8(isInfOrNan) == 0)
            return v;
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(INT32_MIN));
        const __m128i clamped = _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF));
        return _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(isInfOrNan, clamped), _mm_andnot_si128(isInfOrNan, bits)));
    }
#endif
}

bool VU1Interpreter::buildFastPair(const DecodedInstructionPair &decoded,
                                   uint32_t pairIndex,
                                   uint32_t pairCount,
                                   FastPair &out) const
{
    if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved ||
        decoded.mBit || decoded.suppressedLowerVf != 0u)
    {
        return false;
    }

    out = {};
    if (decoded.eBit)
        out.flags |= kFastFlagEBit;
    if (decoded.upperUsage.writesClip && decoded.lowerUsage.readsClip)
        return false;

    if (decoded.upperVfShadowReg != 0u)
    {
        // Reject simultaneous cross-shadow where lower writes a VF register that upper reads.
        const uint8_t lowerWriteReg = decoded.lowerUsage.vfWrite.reg;
        if (lowerWriteReg != 0u && vfReadLanes(decoded.upperUsage, lowerWriteReg) != 0u)
            return false;
        out.flags |= kFastFlagShadowLowerFirst;
    }

    // Decode upper instruction.
    const uint32_t upper = decoded.upper;
    const uint8_t uop = static_cast<uint8_t>(upper & 0x3Fu);
    out.uDest = DEST(upper);
    out.uFt = FT(upper);
    out.uFs = FS(upper);
    out.uFd = FD(upper);

    if (uop <= 0x2Fu)
    {
        if (uop <= 0x03u) { out.upperOp = FU_ADDbc; out.uAux = uop & 3u; }
        else if (uop <= 0x07u) { out.upperOp = FU_SUBbc; out.uAux = uop & 3u; }
        else if (uop <= 0x0Bu) { out.upperOp = FU_MADDbc; out.uAux = uop & 3u; }
        else if (uop <= 0x0Fu) { out.upperOp = FU_MSUBbc; out.uAux = uop & 3u; }
        else if (uop <= 0x13u) { out.upperOp = FU_MAXbc; out.uAux = uop & 3u; }
        else if (uop <= 0x17u) { out.upperOp = FU_MINIbc; out.uAux = uop & 3u; }
        else if (uop <= 0x1Bu) { out.upperOp = FU_MULbc; out.uAux = uop & 3u; }
        else
        {
            switch (uop)
            {
            case 0x1C: out.upperOp = FU_MULq; out.flags |= kFastFlagReadsQ; break;
            case 0x1D: out.upperOp = FU_MAXi; break;
            case 0x1E: out.upperOp = FU_MULi; break;
            case 0x1F: out.upperOp = FU_MINIi; break;
            case 0x20: out.upperOp = FU_ADDq; out.flags |= kFastFlagReadsQ; break;
            case 0x21: out.upperOp = FU_MADDq; out.flags |= kFastFlagReadsQ; break;
            case 0x22: out.upperOp = FU_ADDi; break;
            case 0x23: out.upperOp = FU_MADDi; break;
            case 0x24: out.upperOp = FU_SUBq; out.flags |= kFastFlagReadsQ; break;
            case 0x25: out.upperOp = FU_MSUBq; out.flags |= kFastFlagReadsQ; break;
            case 0x26: out.upperOp = FU_SUBi; break;
            case 0x27: out.upperOp = FU_MSUBi; break;
            case 0x28: out.upperOp = FU_ADD; break;
            case 0x29: out.upperOp = FU_MADD; break;
            case 0x2A: out.upperOp = FU_MUL; break;
            case 0x2B: out.upperOp = FU_MAX; break;
            case 0x2C: out.upperOp = FU_SUB; break;
            case 0x2D: out.upperOp = FU_MSUB; break;
            case 0x2E: out.upperOp = FU_OPMSUB; break;
            case 0x2F: out.upperOp = FU_MINI; break;
            default: return false;
            }
        }
        if (out.uFd == 0u || out.uDest == 0u)
            out.upperOp = FU_NOP;
    }
    else if (uop >= 0x3Cu)
    {
        const uint8_t sp = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        if (sp <= 0x03u) { out.upperOp = FU_ADDAbc; out.uAux = sp & 3u; }
        else if (sp <= 0x07u) { out.upperOp = FU_SUBAbc; out.uAux = sp & 3u; }
        else if (sp <= 0x0Bu) { out.upperOp = FU_MADDAbc; out.uAux = sp & 3u; }
        else if (sp <= 0x0Fu) { out.upperOp = FU_MSUBAbc; out.uAux = sp & 3u; }
        else if (sp >= 0x18u && sp <= 0x1Bu) { out.upperOp = FU_MULAbc; out.uAux = sp & 3u; }
        else
        {
            switch (sp)
            {
            case 0x10: out.upperOp = FU_ITOF0; out.uFd = out.uFt; break;
            case 0x11: out.upperOp = FU_ITOF4; out.uFd = out.uFt; break;
            case 0x12: out.upperOp = FU_ITOF12; out.uFd = out.uFt; break;
            case 0x13: out.upperOp = FU_ITOF15; out.uFd = out.uFt; break;
            case 0x14: out.upperOp = FU_FTOI0; out.uFd = out.uFt; break;
            case 0x15: out.upperOp = FU_FTOI4; out.uFd = out.uFt; break;
            case 0x16: out.upperOp = FU_FTOI12; out.uFd = out.uFt; break;
            case 0x17: out.upperOp = FU_FTOI15; out.uFd = out.uFt; break;
            case 0x1C: out.upperOp = FU_MULAq; out.flags |= kFastFlagReadsQ; break;
            case 0x1D: out.upperOp = FU_ABS; out.uFd = out.uFt; break;
            case 0x1E: out.upperOp = FU_MULAi; break;
            case 0x1F: out.upperOp = FU_CLIPw; break;
            case 0x20: out.upperOp = FU_ADDAq; out.flags |= kFastFlagReadsQ; break;
            case 0x21: out.upperOp = FU_MADDAq; out.flags |= kFastFlagReadsQ; break;
            case 0x22: out.upperOp = FU_ADDAi; break;
            case 0x23: out.upperOp = FU_MADDAi; break;
            case 0x24: out.upperOp = FU_SUBAq; out.flags |= kFastFlagReadsQ; break;
            case 0x25: out.upperOp = FU_MSUBAq; out.flags |= kFastFlagReadsQ; break;
            case 0x26: out.upperOp = FU_SUBAi; break;
            case 0x27: out.upperOp = FU_MSUBAi; break;
            case 0x28: out.upperOp = FU_ADDA; break;
            case 0x29: out.upperOp = FU_MADDA; break;
            case 0x2A: out.upperOp = FU_MULA; break;
            case 0x2C: out.upperOp = FU_SUBA; break;
            case 0x2D: out.upperOp = FU_MSUBA; break;
            case 0x2E: out.upperOp = FU_OPMULA; break;
            case 0x2F:
            case 0x30:
                out.upperOp = FU_NOP;
                break;
            default:
                return false;
            }
        }
        if ((out.upperOp >= FU_ITOF0 && out.upperOp <= FU_ABS) && (out.uFd == 0u || out.uDest == 0u))
            out.upperOp = FU_NOP;
    }
    else
    {
        return false;
    }

    // Reject lower-first shadow pairs where lower updates Q while upper reads Q.
    if ((out.flags & kFastFlagShadowLowerFirst) != 0u &&
        (out.flags & kFastFlagReadsQ) != 0u &&
        decoded.lowerUsage.pipeline == PipelineFdiv)
    {
        return false;
    }

    // Decode lower instruction.
    const uint32_t lower = decoded.lower;
    if (decoded.iBit)
    {
        out.lowerOp = FL_LOI;
        out.imm24 = lower;
        return true;
    }
    if (lower == 0u || lower == 0x8000033Cu)
    {
        out.lowerOp = FL_NOP;
        return true;
    }

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    const uint32_t pcMask = microAddressMask();

    auto setBranchTarget = [&]() -> bool
    {
        const int16_t imm11 = IMM11(lower);
        const uint32_t targetPc = ((pairIndex * 8u) + 8u + static_cast<int32_t>(imm11) * 8) & pcMask;
        const uint32_t targetIdx = targetPc / 8u;
        if (targetIdx >= pairCount)
            return false;
        out.imm = static_cast<int16_t>(targetIdx);
        out.flags |= kFastFlagIsBranch;
        return true;
    };

    switch (opHi)
    {
    case 0x00: out.lowerOp = FL_LQ; out.lReg1 = vfT; out.lReg2 = viS; out.lDest = dest; out.imm = IMM11(lower); return true;
    case 0x01: out.lowerOp = FL_SQ; out.lReg1 = vfS; out.lReg2 = viT; out.lDest = dest; out.imm = IMM11(lower); out.flags |= kFastFlagIsStore; return true;
    case 0x04: out.lowerOp = FL_ILW; out.lReg1 = viT; out.lReg2 = viS; out.lDest = dest; out.imm = IMM11(lower); return true;
    case 0x05: out.lowerOp = FL_ISW; out.lReg1 = viT; out.lReg2 = viS; out.lDest = dest; out.imm = IMM11(lower); out.flags |= kFastFlagIsStore; return true;
    case 0x08: out.lowerOp = FL_IADDIU; out.lReg1 = viT; out.lReg2 = viS; out.imm = static_cast<int16_t>((lower & 0x7FFu) | ((lower >> 10) & 0x7800u)); return true;
    case 0x09: out.lowerOp = FL_ISUBIU; out.lReg1 = viT; out.lReg2 = viS; out.imm = static_cast<int16_t>((lower & 0x7FFu) | ((lower >> 10) & 0x7800u)); return true;
    case 0x10: out.lowerOp = FL_FCEQ; out.imm24 = lower & 0xFFFFFFu; return true;
    case 0x11:
        // FCSET cancels a same-pair CLIP; the fast path applies both at once, so keep that pair on the slow path.
        if (decoded.upperUsage.writesClip)
            return false;
        out.lowerOp = FL_FCSET;
        out.imm24 = lower & 0xFFFFFFu;
        return true;
    case 0x12: out.lowerOp = FL_FCAND; out.imm24 = lower & 0xFFFFFFu; return true;
    case 0x13: out.lowerOp = FL_FCOR; out.imm24 = lower & 0xFFFFFFu; return true;
    case 0x18: out.lowerOp = FL_FMEQ; out.lReg1 = viT; out.lReg2 = viS; return true;
    case 0x1A: out.lowerOp = FL_FMAND; out.lReg1 = viT; out.lReg2 = viS; return true;
    case 0x1B: out.lowerOp = FL_FMOR; out.lReg1 = viT; out.lReg2 = viS; return true;
    case 0x20: out.lowerOp = FL_B; out.flags |= kFastFlagUncondBranch; return setBranchTarget();
    case 0x28: out.lowerOp = FL_IBEQ; out.lReg1 = viS; out.lReg2 = viT; return setBranchTarget();
    case 0x29: out.lowerOp = FL_IBNE; out.lReg1 = viS; out.lReg2 = viT; return setBranchTarget();
    case 0x2C: out.lowerOp = FL_IBLTZ; out.lReg1 = viS; return setBranchTarget();
    case 0x2D: out.lowerOp = FL_IBGTZ; out.lReg1 = viS; return setBranchTarget();
    case 0x2E: out.lowerOp = FL_IBLEZ; out.lReg1 = viS; return setBranchTarget();
    case 0x2F: out.lowerOp = FL_IBGEZ; out.lReg1 = viS; return setBranchTarget();
    case 0x40:
        break;
    default:
        return false;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u) { out.lowerOp = FL_IADD; out.lReg1 = viD; out.lReg2 = viS; out.lReg3 = viT; return true; }
    if (direct == 0x31u) { out.lowerOp = FL_ISUB; out.lReg1 = viD; out.lReg2 = viS; out.lReg3 = viT; return true; }
    if (direct == 0x32u)
    {
        int32_t imm5 = static_cast<int32_t>((lower >> 6) & 0x1Fu);
        if (imm5 & 0x10)
            imm5 |= ~0x1F;
        out.lowerOp = FL_IADDI;
        out.lReg1 = viT;
        out.lReg2 = viS;
        out.imm = static_cast<int16_t>(imm5);
        return true;
    }
    if (direct == 0x34u) { out.lowerOp = FL_IAND; out.lReg1 = viD; out.lReg2 = viS; out.lReg3 = viT; return true; }
    if (direct == 0x35u) { out.lowerOp = FL_IOR; out.lReg1 = viD; out.lReg2 = viS; out.lReg3 = viT; return true; }
    if (direct < 0x3Cu)
        return false;

    const uint8_t sp = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (sp)
    {
    case 0x30:
        out.lowerOp = (vfT == 0u || dest == 0u) ? FL_NOP : FL_MOVE;
        out.lReg1 = vfT;
        out.lReg2 = vfS;
        out.lDest = dest;
        return true;
    case 0x31:
        out.lowerOp = (vfT == 0u || dest == 0u) ? FL_NOP : FL_MR32;
        out.lReg1 = vfT;
        out.lReg2 = vfS;
        out.lDest = dest;
        return true;
    case 0x34: out.lowerOp = FL_LQI; out.lReg1 = vfT; out.lReg2 = viS; out.lDest = dest; return true;
    case 0x35: out.lowerOp = FL_SQI; out.lReg1 = vfS; out.lReg2 = viT; out.lDest = dest; out.flags |= kFastFlagIsStore; return true;
    case 0x36: out.lowerOp = FL_LQD; out.lReg1 = vfT; out.lReg2 = viS; out.lDest = dest; return true;
    case 0x37: out.lowerOp = FL_SQD; out.lReg1 = vfS; out.lReg2 = viT; out.lDest = dest; out.flags |= kFastFlagIsStore; return true;
    case 0x38: out.lowerOp = FL_DIV; out.lReg1 = vfT; out.lReg2 = vfS; out.lDest = static_cast<uint8_t>(((lower >> 21) & 3u) | (((lower >> 23) & 3u) << 2)); return true;
    case 0x39: out.lowerOp = FL_SQRT; out.lReg1 = vfT; out.lDest = static_cast<uint8_t>((lower >> 23) & 3u); return true;
    case 0x3A: out.lowerOp = FL_RSQRT; out.lReg1 = vfT; out.lReg2 = vfS; out.lDest = static_cast<uint8_t>(((lower >> 21) & 3u) | (((lower >> 23) & 3u) << 2)); return true;
    case 0x3B: out.lowerOp = FL_WAITQ; return true;
    case 0x3C: out.lowerOp = FL_MTIR; out.lReg1 = viT; out.lReg2 = vfS; out.lDest = static_cast<uint8_t>((lower >> 21) & 3u); return true;
    case 0x3D: out.lowerOp = FL_MFIR; out.lReg1 = vfT; out.lReg2 = viS; out.lDest = dest; return true;
    case 0x3E: out.lowerOp = FL_ILWR; out.lReg1 = viT; out.lReg2 = viS; out.lDest = dest; return true;
    case 0x3F: out.lowerOp = FL_ISWR; out.lReg1 = viT; out.lReg2 = viS; out.lDest = dest; out.flags |= kFastFlagIsStore; return true;
    case 0x68: out.lowerOp = FL_XTOP; out.lReg1 = viT; return true;
    case 0x69: out.lowerOp = FL_XITOP; out.lReg1 = viT; return true;
    case 0x6C: out.lowerOp = FL_XGKICK; out.lReg1 = viS; return true;
    default:
        return false;
    }
}

bool VU1Interpreter::isFastEntryEligible(uint32_t entryPair, uint32_t pairCount)
{
    if (entryPair >= pairCount)
        return false;
    const uint8_t cached = m_fastEntryStatus[entryPair];
    if (cached != 0u)
        return cached == 1u;

    m_fastEntryStatus[entryPair] = 2u;

    // Pass 1: Reachability BFS + FastPair lowering + CFG edges into a staged table
    // so a rejected or conflicting entry never mutates already-committed entries.
    std::vector<FastPair> staged(pairCount);
    std::array<uint8_t, kMaxDecodedPairs> visited{};
    std::array<uint8_t, kMaxDecodedPairs> isDelaySlot{};
    std::array<uint8_t, kMaxDecodedPairs> isTerminalDelay{};
    std::array<uint8_t, kMaxDecodedPairs> inDegree{};
    std::array<uint8_t, kMaxDecodedPairs> succCount{};
    std::array<std::array<uint16_t, 2>, kMaxDecodedPairs> succ{};
    std::array<uint16_t, kMaxDecodedPairs> queue{};
    std::array<uint16_t, kMaxDecodedPairs> reachableList{};
    std::array<uint16_t, kMaxDecodedPairs> delayTaken{};
    std::array<uint16_t, kMaxDecodedPairs> delayFall{};
    uint32_t qHead = 0u, qTail = 0u, reachableCount = 0u;

    auto enqueue = [&](uint32_t idx) -> bool
    {
        if (idx >= pairCount)
            return false;
        if (!visited[idx])
        {
            visited[idx] = 1u;
            queue[qTail++] = static_cast<uint16_t>(idx);
            reachableList[reachableCount++] = static_cast<uint16_t>(idx);
        }
        return true;
    };

    auto addEdge = [&](uint32_t from, uint32_t to) -> bool
    {
        if (to >= pairCount || succCount[from] >= 2u)
            return false;
        for (uint8_t k = 0u; k < succCount[from]; ++k)
            if (succ[from][k] == to)
                return true;
        succ[from][succCount[from]++] = static_cast<uint16_t>(to);
        if (inDegree[to] < 255u)
            ++inDegree[to];
        return enqueue(to);
    };

    enqueue(entryPair);
    bool hasXgkick = false;
    bool hasEBit = false;

    while (qHead < qTail)
    {
        const uint32_t idx = queue[qHead++];
        const DecodedInstructionPair &d = m_decodedCodeCache[idx];
        FastPair fp{};
        if (!buildFastPair(d, idx, pairCount, fp))
            return false;
        staged[idx] = fp;

        if (fp.lowerOp == FL_XGKICK)
            hasXgkick = true;

        const uint32_t nextIdx = (idx + 1u < pairCount) ? (idx + 1u) : 0u;
        if (isDelaySlot[idx])
        {
            if ((fp.flags & (kFastFlagIsBranch | kFastFlagEBit)) != 0u || fp.lowerOp == FL_XGKICK)
                return false;
            continue;
        }

        if ((fp.flags & kFastFlagEBit) != 0u)
        {
            hasEBit = true;
            if ((fp.flags & kFastFlagIsBranch) != 0u)
                return false;
            isDelaySlot[nextIdx] = 1u;
            isTerminalDelay[nextIdx] = 1u;
            if (!addEdge(idx, nextIdx))
                return false;
            continue;
        }

        if ((fp.flags & kFastFlagIsBranch) != 0u)
        {
            const uint32_t delayIdx = nextIdx;
            const uint32_t fallIdx = (delayIdx + 1u < pairCount) ? (delayIdx + 1u) : 0u;
            const uint32_t targetIdx = static_cast<uint32_t>(fp.imm);
            isDelaySlot[delayIdx] = 1u;
            delayTaken[delayIdx] = static_cast<uint16_t>(targetIdx);
            delayFall[delayIdx] = static_cast<uint16_t>((fp.flags & kFastFlagUncondBranch) != 0u ? targetIdx : fallIdx);
            if (!addEdge(idx, delayIdx))
                return false;
            if (!addEdge(delayIdx, targetIdx))
                return false;
            if ((fp.flags & kFastFlagUncondBranch) == 0u)
            {
                if (!addEdge(delayIdx, fallIdx))
                    return false;
            }
            continue;
        }

        if (!addEdge(idx, nextIdx))
            return false;
    }

    if (!hasXgkick || !hasEBit)
        return false;

    // Delay slots must have unique entry from their branch/E-bit pair.
    for (uint32_t r = 0u; r < reachableCount; ++r)
    {
        const uint32_t idx = reachableList[r];
        if (isDelaySlot[idx] && inDegree[idx] != 1u)
            return false;
    }

    // Verify every XGKICK is strictly terminal (cannot reach another XGKICK or any store, and reaches E-bit).
    for (uint32_t r = 0u; r < reachableCount; ++r)
    {
        const uint32_t startIdx = reachableList[r];
        if (staged[startIdx].lowerOp != FL_XGKICK)
            continue;
        std::array<uint8_t, kMaxDecodedPairs> seenAfter{};
        uint32_t dHead = 0u, dTail = 0u;
        for (uint8_t k = 0u; k < succCount[startIdx]; ++k)
        {
            const uint16_t s = succ[startIdx][k];
            seenAfter[s] = 1u;
            queue[dTail++] = s;
        }
        while (dHead < dTail)
        {
            const uint32_t cur = queue[dHead++];
            const FastPair &cfp = staged[cur];
            if (cfp.lowerOp == FL_XGKICK || (cfp.flags & kFastFlagIsStore) != 0u)
                return false;
            if (cur == startIdx)
                return false;
            for (uint8_t k = 0u; k < succCount[cur]; ++k)
            {
                const uint16_t s = succ[cur][k];
                if (!seenAfter[s])
                {
                    seenAfter[s] = 1u;
                    queue[dTail++] = s;
                }
            }
        }
    }

    // Pass 2: walk (pair, pending-latency state) nodes from the entry. A pair reached under different pending
    // latencies gets its own node per state, so every node's stall and tail latency are exact on every path.
    struct RemState
    {
        uint8_t vf[32][4]{};
        uint8_t vi[16]{};
        uint8_t clip = 0u;
        uint8_t fdiv = 0u;
        uint16_t branchDelayViMask = 0u;
    };
    static_assert(std::has_unique_object_representations_v<RemState>, "states are compared with memcmp");

    constexpr uint32_t kNoNode = 0xFFFFFFFFu;
    const uint32_t maxNodes = reachableCount * 16u + 256u;
    std::vector<FastNode> nodes;
    std::vector<RemState> nodeIn;
    std::vector<uint32_t> nextAtPair;
    std::vector<uint32_t> firstAtPair(pairCount, kNoNode);
    std::vector<uint32_t> work;

    auto nodeFor = [&](uint32_t pair, const RemState &st) -> uint32_t
    {
        for (uint32_t n = firstAtPair[pair]; n != kNoNode; n = nextAtPair[n])
            if (std::memcmp(&nodeIn[n], &st, sizeof(RemState)) == 0)
                return n;
        if (nodes.size() >= maxNodes)
            return kNoNode;
        const uint32_t n = static_cast<uint32_t>(nodes.size());
        FastNode node{};
        node.next = kNoNode;
        node.nextTaken = kNoNode;
        node.pair = static_cast<uint16_t>(pair);
        nodes.push_back(node);
        nodeIn.push_back(st);
        nextAtPair.push_back(firstAtPair[pair]);
        firstAtPair[pair] = n;
        work.push_back(n);
        return n;
    };

    auto advanceRem = [](RemState &s, uint8_t delta)
    {
        if (delta == 0u)
            return;
        for (uint32_t r = 1u; r < 32u; ++r)
            for (uint32_t c = 0u; c < 4u; ++c)
                s.vf[r][c] = s.vf[r][c] > delta ? static_cast<uint8_t>(s.vf[r][c] - delta) : 0u;
        for (uint32_t r = 1u; r < 16u; ++r)
            s.vi[r] = s.vi[r] > delta ? static_cast<uint8_t>(s.vi[r] - delta) : 0u;
        s.clip = s.clip > delta ? static_cast<uint8_t>(s.clip - delta) : 0u;
        s.fdiv = s.fdiv > delta ? static_cast<uint8_t>(s.fdiv - delta) : 0u;
    };

    nodeFor(entryPair, RemState{});
    bool readsMac = false;
    while (!work.empty())
    {
        const uint32_t n = work.back();
        work.pop_back();
        const uint32_t idx = nodes[n].pair;
        const DecodedInstructionPair &d = m_decodedCodeCache[idx];
        const FastPair &fp = staged[idx];
        RemState st = nodeIn[n];

        if (fp.lowerOp == FL_FMEQ || fp.lowerOp == FL_FMAND || fp.lowerOp == FL_FMOR)
            readsMac = true;
        if ((fp.flags & kFastFlagIsBranch) != 0u && (d.lowerUsage.viRead & st.branchDelayViMask) != 0u)
            return false;

        // Compute stall cycles matching calculatePairReadyCycle.
        uint8_t stall = 0u;
        const InstructionUsage *usages[2] = {&d.upperUsage, &d.lowerUsage};
        for (const InstructionUsage *u : usages)
        {
            for (uint32_t i = 0u; i < u->vfReadCount; ++i)
            {
                const VfAccess &acc = u->vfRead[i];
                for (uint32_t c = 0u; c < 4u; ++c)
                {
                    if ((acc.lanes & laneForComponent(c)) != 0u)
                        stall = std::max(stall, st.vf[acc.reg][c]);
                }
            }
            for (uint32_t pending = u->viRead & ~1u; pending != 0u; pending &= pending - 1u)
                stall = std::max(stall, st.vi[std::countr_zero(pending)]);
        }
        if (d.lowerUsage.pipeline == PipelineFdiv || d.lowerUsage.waitQ)
            stall = std::max(stall, st.fdiv);
        advanceRem(st, stall);
        nodes[n].cycleStep = static_cast<uint8_t>(1u + stall);

        // Non-stalling reads see only committed CLIP and Q, which the fast path applies at issue.
        if (d.lowerUsage.readsClip && st.clip > 0u)
            return false;
        if ((fp.flags & kFastFlagReadsQ) != 0u && st.fdiv > 0u)
            return false;

        // Check WAW hazards (slower earlier write would overwrite faster later write).
        auto applyVfWrite = [&](const InstructionUsage &u) -> bool
        {
            const VfAccess &w = u.vfWrite;
            if (w.reg == 0u || w.lanes == 0u)
                return true;
            const uint8_t lat = u.vfLatency != 0u ? u.vfLatency : u.latency;
            for (uint32_t c = 0u; c < 4u; ++c)
            {
                if ((w.lanes & laneForComponent(c)) != 0u)
                {
                    if (st.vf[w.reg][c] > lat)
                        return false;
                    st.vf[w.reg][c] = lat;
                }
            }
            return true;
        };

        if (!applyVfWrite(d.lowerUsage))
            return false;
        if (!applyVfWrite(d.upperUsage))
            return false;

        if ((d.lowerUsage.viWrite & ~1u) != 0u)
        {
            const uint8_t lat = d.lowerUsage.viLatency != 0u ? d.lowerUsage.viLatency : d.lowerUsage.latency;
            for (uint32_t w = d.lowerUsage.viWrite & ~1u; w != 0u; w &= w - 1u)
            {
                const uint32_t reg = std::countr_zero(w);
                if (st.vi[reg] > lat)
                    return false;
                st.vi[reg] = lat;
            }
        }
        if (d.upperUsage.writesClip || d.lowerUsage.writesClip)
            st.clip = kFmacLatency;
        if (d.lowerUsage.pipeline == PipelineFdiv)
            st.fdiv = d.lowerUsage.latency;

        st.branchDelayViMask = d.lowerUsage.delaysNextBranchRead ? static_cast<uint16_t>(d.lowerUsage.viWrite & ~1u) : 0u;

        // Advance 1 cycle at end of pair.
        advanceRem(st, 1u);

        if (isTerminalDelay[idx])
        {
            uint8_t maxRem = std::max(st.clip, st.fdiv);
            for (uint32_t r = 1u; r < 32u; ++r)
                for (uint32_t c = 0u; c < 4u; ++c)
                    maxRem = std::max(maxRem, st.vf[r][c]);
            for (uint32_t r = 1u; r < 16u; ++r)
                maxRem = std::max(maxRem, st.vi[r]);
            nodes[n].tailLatency = maxRem;
            continue;
        }

        if (isDelaySlot[idx])
        {
            const uint32_t taken = nodeFor(delayTaken[idx], st);
            const uint32_t fall = nodeFor(delayFall[idx], st);
            if (taken == kNoNode || fall == kNoNode)
                return false;
            nodes[n].nextTaken = taken;
            nodes[n].next = fall;
        }
        else
        {
            const uint32_t next = nodeFor(succ[idx][0], st);
            if (next == kNoNode)
                return false;
            nodes[n].next = next;
        }
    }

    // A MAC read (FMEQ/FMAND/FMOR) sees the newest FMAC result issued at least kFmacLatency cycles earlier. Walk
    // back from each reader along every path and mark that producer: only marked nodes compute exact MAC flags.
    if (readsMac)
    {
        auto writesMac = [&](uint32_t pair)
        {
            const uint32_t upper = m_decodedCodeCache[pair].upper;
            const uint32_t op = upper & 0x3Fu;
            const uint32_t a = op >= 0x3Cu ? ((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
            return DEST(upper) != 0u &&
                   (a <= 0x0Fu || (a >= 0x18u && a <= 0x1Cu) || a == 0x1Eu || (a >= 0x20u && a <= 0x2Eu && a != 0x2Bu));
        };

        std::vector<std::vector<uint32_t>> preds(nodes.size());
        for (uint32_t n = 0u; n < nodes.size(); ++n)
        {
            if (nodes[n].next != kNoNode)
                preds[nodes[n].next].push_back(n);
            if (nodes[n].nextTaken != kNoNode && nodes[n].nextTaken != nodes[n].next)
                preds[nodes[n].nextTaken].push_back(n);
        }

        for (uint32_t r = 0u; r < nodes.size(); ++r)
        {
            const uint8_t op = staged[nodes[r].pair].lowerOp;
            if (op != FL_FMEQ && op != FL_FMAND && op != FL_FMOR)
                continue;
            // The run's first pair sees the MAC from before the run, which the fast path does not keep.
            if (r == 0u)
                return false;
            std::vector<uint8_t> seen(nodes.size() * kFmacLatency + nodes.size());
            std::vector<std::pair<uint32_t, uint32_t>> stack;
            for (uint32_t p : preds[r])
                stack.emplace_back(p, std::min<uint32_t>(nodes[r].cycleStep, kFmacLatency));
            while (!stack.empty())
            {
                const auto [x, dist] = stack.back();
                stack.pop_back();
                uint8_t &mark = seen[x * (kFmacLatency + 1u) + dist];
                if (mark)
                    continue;
                mark = 1u;
                if (dist >= kFmacLatency && writesMac(nodes[x].pair))
                {
                    nodes[x].macProducer = true;
                    continue;
                }
                if (x == 0u)
                    return false;
                for (uint32_t p : preds[x])
                    stack.emplace_back(p, std::min<uint32_t>(dist + nodes[x].cycleStep, kFmacLatency));
            }
        }
    }

    const uint32_t base = static_cast<uint32_t>(m_fastNodes.size());
    for (FastNode node : nodes)
    {
        if (node.next != kNoNode)
            node.next += base;
        if (node.nextTaken != kNoNode)
            node.nextTaken += base;
        m_fastNodes.push_back(node);
    }
    for (uint32_t r = 0u; r < reachableCount; ++r)
        m_fastCodeCache[reachableList[r]] = staged[reachableList[r]];

    m_fastEntryNode[entryPair] = base;
    m_fastEntryStatus[entryPair] = 1u;
    return true;
}

// shortcut: SSE2 fast path assumes finite VF inputs under RZ+FTZ+DAZ (skipping per-op Inf/NaN clamp and dead MAC/STATUS flags);
// upgrade with sseClampVuInf on FMAC ops if a game feeds raw Inf/NaN bit patterns into eligible VU1 loops.
void VU1Interpreter::runFastDirect(uint8_t *vuData, uint32_t dataSize, uint32_t pairCount, uint64_t budgetEnd)
{
#if defined(__SSE2__) || defined(_M_X64)
    ScopedVuMxcsr mxcsrGuard;
    alignas(16) __m128 vf[32];
    for (uint32_t r = 0u; r < 32u; ++r)
        vf[r] = _mm_loadu_ps(m_state.vf[r]);
    vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    __m128 acc = _mm_loadu_ps(m_state.acc);

    auto getLaneF = [&](uint8_t reg, uint8_t lane) -> float
    {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, vf[reg]);
        return tmp[lane];
    };
    auto getLaneU32 = [&](uint8_t reg, uint8_t lane) -> uint32_t
    {
        alignas(16) uint32_t tmp[4];
        _mm_store_ps(reinterpret_cast<float *>(tmp), vf[reg]);
        return tmp[lane];
    };
#else
    alignas(16) float vf[32][4];
    std::memcpy(vf, m_state.vf, sizeof(vf));
    vf[0][0] = 0.0f; vf[0][1] = 0.0f; vf[0][2] = 0.0f; vf[0][3] = 1.0f;
    float acc[4];
    std::memcpy(acc, m_state.acc, sizeof(acc));

    auto getLaneF = [&](uint8_t reg, uint8_t lane) -> float { return vf[reg][lane]; };
    auto getLaneU32 = [&](uint8_t reg, uint8_t lane) -> uint32_t { return std::bit_cast<uint32_t>(vf[reg][lane]); };
#endif

    int32_t vi[16];
    std::memcpy(vi, m_state.vi, sizeof(vi));
    vi[0] = 0;

    float q = m_state.q;
    float iReg = m_state.i;
    uint32_t clip = m_state.clip;
    const uint32_t dataMask = dataSize - 1u;
    uint64_t cycles = m_cycle;
    uint32_t pcIdx = m_state.pc / 8u;
    const FastPair *fast = m_fastCodeCache.data();
    const FastNode *nodes = m_fastNodes.data();
    uint32_t nodeIdx = m_fastEntryNode[pcIdx];
    uint64_t issueCycle = cycles; // cycle the current pair issues on, after its stall

    // MAC results of the producer nodes, newest last; a reader takes the newest one already through the pipeline.
    // shortcut: unmarked FMACs keep no flags and m_state.mac is not written back, so a slow run that reads MAC
    // before its first FMAC sees a stale value. Keeping the last FMAC's operands to compute it at the end cost
    // 0.2 ms per field on the battle models (2026-11-27); do it if such a reader shows up.
    std::array<uint64_t, 8> macIssue{};
    std::array<uint32_t, 8> macValue{};
    uint32_t macCount = 0u;

    bool xgkickIssued = false;
    uint64_t xgkickIssueCycle = 0u;
    uint32_t xgkickSourceQword = 0u;

    auto execFastUpper = [&](const FastPair &p)
    {
#if defined(__SSE2__) || defined(_M_X64)
        switch (p.upperOp)
        {
        case FU_NOP:
            return;
        case FU_MULAbc:
            sseApplyMask(acc, _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_MADDAbc:
            sseApplyMask(acc, _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux))), p.uDest);
            return;
        case FU_MADDbc:
            sseApplyMask(vf[p.uFd], _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux))), p.uDest);
            return;
        case FU_MSUBbc:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux))), p.uDest);
            return;
        case FU_MSUBAbc:
            sseApplyMask(acc, _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux))), p.uDest);
            return;
        case FU_MULbc:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_ADDbc:
            sseApplyMask(vf[p.uFd], _mm_add_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_SUBbc:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_ADDAbc:
            sseApplyMask(acc, _mm_add_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_SUBAbc:
            sseApplyMask(acc, _mm_sub_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_MAXbc:
            sseApplyMask(vf[p.uFd], _mm_max_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_MINIbc:
            sseApplyMask(vf[p.uFd], _mm_min_ps(vf[p.uFs], sseBroadcastLane(vf[p.uFt], p.uAux)), p.uDest);
            return;
        case FU_MULq:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_MULi:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_ADDq:
            sseApplyMask(vf[p.uFd], _mm_add_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_ADDi:
            sseApplyMask(vf[p.uFd], _mm_add_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_SUBq:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_SUBi:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_MADDq:
            sseApplyMask(vf[p.uFd], _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q))), p.uDest);
            return;
        case FU_MADDi:
            sseApplyMask(vf[p.uFd], _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg))), p.uDest);
            return;
        case FU_MSUBq:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q))), p.uDest);
            return;
        case FU_MSUBi:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg))), p.uDest);
            return;
        case FU_MAXi:
            sseApplyMask(vf[p.uFd], _mm_max_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_MINIi:
            sseApplyMask(vf[p.uFd], _mm_min_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_ADD:
            sseApplyMask(vf[p.uFd], _mm_add_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_SUB:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_MUL:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_MADD:
            sseApplyMask(vf[p.uFd], _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], vf[p.uFt])), p.uDest);
            return;
        case FU_MSUB:
            sseApplyMask(vf[p.uFd], _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], vf[p.uFt])), p.uDest);
            return;
        case FU_MAX:
            sseApplyMask(vf[p.uFd], _mm_max_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_MINI:
            sseApplyMask(vf[p.uFd], _mm_min_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_ADDA:
            sseApplyMask(acc, _mm_add_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_SUBA:
            sseApplyMask(acc, _mm_sub_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_MULA:
            sseApplyMask(acc, _mm_mul_ps(vf[p.uFs], vf[p.uFt]), p.uDest);
            return;
        case FU_MADDA:
            sseApplyMask(acc, _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], vf[p.uFt])), p.uDest);
            return;
        case FU_MSUBA:
            sseApplyMask(acc, _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], vf[p.uFt])), p.uDest);
            return;
        case FU_MULAq:
            sseApplyMask(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_MULAi:
            sseApplyMask(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_ADDAq:
            sseApplyMask(acc, _mm_add_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_ADDAi:
            sseApplyMask(acc, _mm_add_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_SUBAq:
            sseApplyMask(acc, _mm_sub_ps(vf[p.uFs], _mm_set1_ps(q)), p.uDest);
            return;
        case FU_SUBAi:
            sseApplyMask(acc, _mm_sub_ps(vf[p.uFs], _mm_set1_ps(iReg)), p.uDest);
            return;
        case FU_MADDAq:
            sseApplyMask(acc, _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q))), p.uDest);
            return;
        case FU_MADDAi:
            sseApplyMask(acc, _mm_add_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg))), p.uDest);
            return;
        case FU_MSUBAq:
            sseApplyMask(acc, _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(q))), p.uDest);
            return;
        case FU_MSUBAi:
            sseApplyMask(acc, _mm_sub_ps(acc, _mm_mul_ps(vf[p.uFs], _mm_set1_ps(iReg))), p.uDest);
            return;
        case FU_OPMULA:
        {
            const __m128 a = _mm_shuffle_ps(vf[p.uFs], vf[p.uFs], _MM_SHUFFLE(3, 0, 2, 1));
            const __m128 b = _mm_shuffle_ps(vf[p.uFt], vf[p.uFt], _MM_SHUFFLE(3, 1, 0, 2));
            sseApplyMask(acc, _mm_mul_ps(a, b), p.uDest & 0xEu);
            if (p.uDest & 1u)
                sseApplyMask(acc, _mm_setzero_ps(), 1u);
            return;
        }
        case FU_OPMSUB:
        {
            const __m128 a = _mm_shuffle_ps(vf[p.uFs], vf[p.uFs], _MM_SHUFFLE(3, 0, 2, 1));
            const __m128 b = _mm_shuffle_ps(vf[p.uFt], vf[p.uFt], _MM_SHUFFLE(3, 1, 0, 2));
            sseApplyMask(vf[p.uFd], _mm_sub_ps(acc, _mm_mul_ps(a, b)), p.uDest & 0xEu);
            if (p.uDest & 1u)
                sseApplyMask(vf[p.uFd], _mm_setzero_ps(), 1u);
            return;
        }
        case FU_ABS:
        {
            const __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
            sseApplyMask(vf[p.uFd], _mm_and_ps(sseClampVuInf(vf[p.uFs]), absMask), p.uDest);
            return;
        }
        case FU_ITOF0:
            sseApplyMask(vf[p.uFd], _mm_cvtepi32_ps(_mm_castps_si128(vf[p.uFs])), p.uDest);
            return;
        case FU_ITOF4:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vf[p.uFs])), _mm_set1_ps(1.0f / 16.0f)), p.uDest);
            return;
        case FU_ITOF12:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vf[p.uFs])), _mm_set1_ps(1.0f / 4096.0f)), p.uDest);
            return;
        case FU_ITOF15:
            sseApplyMask(vf[p.uFd], _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vf[p.uFs])), _mm_set1_ps(1.0f / 32768.0f)), p.uDest);
            return;
        case FU_FTOI0:
        case FU_FTOI4:
        case FU_FTOI12:
        case FU_FTOI15:
        {
            const float scale = (p.upperOp == FU_FTOI0) ? 1.0f
                              : (p.upperOp == FU_FTOI4) ? 16.0f
                              : (p.upperOp == FU_FTOI12) ? 4096.0f : 32768.0f;
            alignas(16) float src[4];
            alignas(16) int32_t outI[4]{};
            _mm_store_ps(src, vf[p.uFs]);
            for (uint32_t c = 0u; c < 4u; ++c)
                if ((p.uDest & (8u >> c)) != 0u)
                    outI[c] = fastVuFloatToInt(normalizeOperand(src[c]), scale);
            sseApplyMask(vf[p.uFd], _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(outI))), p.uDest);
            return;
        }
        case FU_CLIPw:
        {
            alignas(16) uint32_t sBits[4];
            _mm_store_ps(reinterpret_cast<float *>(sBits), vf[p.uFs]);
            const uint32_t wBits = getLaneU32(p.uFt, 3u);
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
            uint32_t flags = 0u;
            if (static_cast<int32_t>(sBits[0]) > limit) flags |= 0x01u;
            if (static_cast<int32_t>(sBits[0] ^ 0x80000000u) > limit) flags |= 0x02u;
            if (static_cast<int32_t>(sBits[1]) > limit) flags |= 0x04u;
            if (static_cast<int32_t>(sBits[1] ^ 0x80000000u) > limit) flags |= 0x08u;
            if (static_cast<int32_t>(sBits[2]) > limit) flags |= 0x10u;
            if (static_cast<int32_t>(sBits[2] ^ 0x80000000u) > limit) flags |= 0x20u;
            clip = ((clip << 6u) | flags) & 0x00FFFFFFu;
            return;
        }
        default:
            return;
        }
#else
        auto applyVec = [&](float dst[4], const float res[4], uint8_t mask)
        {
            if (mask & 8u) dst[0] = res[0];
            if (mask & 4u) dst[1] = res[1];
            if (mask & 2u) dst[2] = res[2];
            if (mask & 1u) dst[3] = res[3];
        };
        float res[4]{};
        const float *vs = vf[p.uFs];
        const float *vt = vf[p.uFt];
        switch (p.upperOp)
        {
        case FU_NOP: return;
        case FU_MULAbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] * bc; applyVec(acc, res, p.uDest); return; }
        case FU_MADDAbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * bc; applyVec(acc, res, p.uDest); return; }
        case FU_MADDbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_MSUBbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_MSUBAbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * bc; applyVec(acc, res, p.uDest); return; }
        case FU_MULbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] * bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_ADDbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] + bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_SUBbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] - bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_ADDAbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] + bc; applyVec(acc, res, p.uDest); return; }
        case FU_SUBAbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = vs[c] - bc; applyVec(acc, res, p.uDest); return; }
        case FU_MAXbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = (vs[c] > bc) ? vs[c] : bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_MINIbc: { const float bc = vt[p.uAux]; for (int c = 0; c < 4; ++c) res[c] = (vs[c] < bc) ? vs[c] : bc; applyVec(vf[p.uFd], res, p.uDest); return; }
        case FU_MULq: for (int c = 0; c < 4; ++c) res[c] = vs[c] * q; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MULi: for (int c = 0; c < 4; ++c) res[c] = vs[c] * iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ADDq: for (int c = 0; c < 4; ++c) res[c] = vs[c] + q; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ADDi: for (int c = 0; c < 4; ++c) res[c] = vs[c] + iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_SUBq: for (int c = 0; c < 4; ++c) res[c] = vs[c] - q; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_SUBi: for (int c = 0; c < 4; ++c) res[c] = vs[c] - iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MADDq: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * q; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MADDi: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MSUBq: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * q; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MSUBi: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MAXi: for (int c = 0; c < 4; ++c) res[c] = (vs[c] > iReg) ? vs[c] : iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MINIi: for (int c = 0; c < 4; ++c) res[c] = (vs[c] < iReg) ? vs[c] : iReg; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ADD: for (int c = 0; c < 4; ++c) res[c] = vs[c] + vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_SUB: for (int c = 0; c < 4; ++c) res[c] = vs[c] - vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MUL: for (int c = 0; c < 4; ++c) res[c] = vs[c] * vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MADD: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MSUB: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MAX: for (int c = 0; c < 4; ++c) res[c] = (vs[c] > vt[c]) ? vs[c] : vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_MINI: for (int c = 0; c < 4; ++c) res[c] = (vs[c] < vt[c]) ? vs[c] : vt[c]; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ADDA: for (int c = 0; c < 4; ++c) res[c] = vs[c] + vt[c]; applyVec(acc, res, p.uDest); return;
        case FU_SUBA: for (int c = 0; c < 4; ++c) res[c] = vs[c] - vt[c]; applyVec(acc, res, p.uDest); return;
        case FU_MULA: for (int c = 0; c < 4; ++c) res[c] = vs[c] * vt[c]; applyVec(acc, res, p.uDest); return;
        case FU_MADDA: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * vt[c]; applyVec(acc, res, p.uDest); return;
        case FU_MSUBA: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * vt[c]; applyVec(acc, res, p.uDest); return;
        case FU_MULAq: for (int c = 0; c < 4; ++c) res[c] = vs[c] * q; applyVec(acc, res, p.uDest); return;
        case FU_MULAi: for (int c = 0; c < 4; ++c) res[c] = vs[c] * iReg; applyVec(acc, res, p.uDest); return;
        case FU_ADDAq: for (int c = 0; c < 4; ++c) res[c] = vs[c] + q; applyVec(acc, res, p.uDest); return;
        case FU_ADDAi: for (int c = 0; c < 4; ++c) res[c] = vs[c] + iReg; applyVec(acc, res, p.uDest); return;
        case FU_SUBAq: for (int c = 0; c < 4; ++c) res[c] = vs[c] - q; applyVec(acc, res, p.uDest); return;
        case FU_SUBAi: for (int c = 0; c < 4; ++c) res[c] = vs[c] - iReg; applyVec(acc, res, p.uDest); return;
        case FU_MADDAq: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * q; applyVec(acc, res, p.uDest); return;
        case FU_MADDAi: for (int c = 0; c < 4; ++c) res[c] = acc[c] + vs[c] * iReg; applyVec(acc, res, p.uDest); return;
        case FU_MSUBAq: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * q; applyVec(acc, res, p.uDest); return;
        case FU_MSUBAi: for (int c = 0; c < 4; ++c) res[c] = acc[c] - vs[c] * iReg; applyVec(acc, res, p.uDest); return;
        case FU_OPMULA: res[0] = vs[1] * vt[2]; res[1] = vs[2] * vt[0]; res[2] = vs[0] * vt[1]; res[3] = 0.0f; applyVec(acc, res, p.uDest); return;
        case FU_OPMSUB: res[0] = acc[0] - vs[1] * vt[2]; res[1] = acc[1] - vs[2] * vt[0]; res[2] = acc[2] - vs[0] * vt[1]; res[3] = 0.0f; applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ABS: for (int c = 0; c < 4; ++c) res[c] = std::fabs(normalizeOperand(vs[c])); applyVec(vf[p.uFd], res, p.uDest); return;
        case FU_ITOF0: case FU_ITOF4: case FU_ITOF12: case FU_ITOF15: {
            const float inv = (p.upperOp == FU_ITOF0) ? 1.0f : (p.upperOp == FU_ITOF4) ? (1.0f / 16.0f) : (p.upperOp == FU_ITOF12) ? (1.0f / 4096.0f) : (1.0f / 32768.0f);
            for (int c = 0; c < 4; ++c) res[c] = static_cast<float>(std::bit_cast<int32_t>(vs[c])) * inv;
            applyVec(vf[p.uFd], res, p.uDest); return;
        }
        case FU_FTOI0: case FU_FTOI4: case FU_FTOI12: case FU_FTOI15: {
            const float scale = (p.upperOp == FU_FTOI0) ? 1.0f : (p.upperOp == FU_FTOI4) ? 16.0f : (p.upperOp == FU_FTOI12) ? 4096.0f : 32768.0f;
            for (int c = 0; c < 4; ++c) if ((p.uDest & (8u >> c)) != 0u) res[c] = std::bit_cast<float>(fastVuFloatToInt(normalizeOperand(vs[c]), scale));
            applyVec(vf[p.uFd], res, p.uDest); return;
        }
        case FU_CLIPw: {
            const uint32_t wBits = std::bit_cast<uint32_t>(vt[3]);
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
            uint32_t flags = 0u;
            const uint32_t bx = std::bit_cast<uint32_t>(vs[0]);
            const uint32_t by = std::bit_cast<uint32_t>(vs[1]);
            const uint32_t bz = std::bit_cast<uint32_t>(vs[2]);
            if (static_cast<int32_t>(bx) > limit) flags |= 0x01u;
            if (static_cast<int32_t>(bx ^ 0x80000000u) > limit) flags |= 0x02u;
            if (static_cast<int32_t>(by) > limit) flags |= 0x04u;
            if (static_cast<int32_t>(by ^ 0x80000000u) > limit) flags |= 0x08u;
            if (static_cast<int32_t>(bz) > limit) flags |= 0x10u;
            if (static_cast<int32_t>(bz ^ 0x80000000u) > limit) flags |= 0x20u;
            clip = ((clip << 6u) | flags) & 0x00FFFFFFu;
            return;
        }
        default: return;
        }
#endif
    };

    auto storeVfToMem = [&](uint32_t addr, uint8_t reg, uint8_t dest)
    {
        if (addr + 16u > dataSize)
            return;
#if defined(__SSE2__) || defined(_M_X64)
        if (dest == 0xFu)
        {
            _mm_storeu_ps(reinterpret_cast<float *>(vuData + addr), vf[reg]);
            return;
        }
        const __m128 cur = _mm_loadu_ps(reinterpret_cast<const float *>(vuData + addr));
        const __m128 mask = _mm_load_ps(reinterpret_cast<const float *>(kFastLaneMasks[dest]));
        _mm_storeu_ps(reinterpret_cast<float *>(vuData + addr), _mm_or_ps(_mm_and_ps(mask, vf[reg]), _mm_andnot_ps(mask, cur)));
#else
        for (uint32_t c = 0u; c < 4u; ++c)
            if ((dest & (8u >> c)) != 0u)
                std::memcpy(vuData + addr + c * 4u, &vf[reg][c], 4u);
#endif
    };

    auto loadVfFromMem = [&](uint32_t addr, uint8_t reg, uint8_t dest)
    {
        if (reg == 0u || dest == 0u || addr + 16u > dataSize)
            return;
#if defined(__SSE2__) || defined(_M_X64)
        const __m128 loaded = _mm_loadu_ps(reinterpret_cast<const float *>(vuData + addr));
        sseApplyMask(vf[reg], loaded, dest);
#else
        for (uint32_t c = 0u; c < 4u; ++c)
            if ((dest & (8u >> c)) != 0u)
                std::memcpy(&vf[reg][c], vuData + addr + c * 4u, 4u);
#endif
    };

    auto execFastLower = [&](const FastPair &p) -> bool
    {
        switch (p.lowerOp)
        {
        case FL_NOP:
            return false;
        case FL_LOI:
            iReg = normalizeOperand(std::bit_cast<float>(p.imm24));
            return false;
        case FL_LQ:
        {
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2] + p.imm) * 16u) & dataMask;
            loadVfFromMem(addr, p.lReg1, p.lDest);
            return false;
        }
        case FL_SQ:
        {
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2] + p.imm) * 16u) & dataMask;
            storeVfToMem(addr, p.lReg1, p.lDest);
            return false;
        }
        case FL_LQI:
        {
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2]) * 16u) & dataMask;
            loadVfFromMem(addr, p.lReg1, p.lDest);
            if (p.lReg2 != 0u)
                vi[p.lReg2] = static_cast<int16_t>(vi[p.lReg2] + 1);
            return false;
        }
        case FL_SQI:
        {
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2]) * 16u) & dataMask;
            storeVfToMem(addr, p.lReg1, p.lDest);
            if (p.lReg2 != 0u)
                vi[p.lReg2] = static_cast<int16_t>(vi[p.lReg2] + 1);
            return false;
        }
        case FL_LQD:
        {
            if (p.lReg2 != 0u)
                vi[p.lReg2] = static_cast<int16_t>(vi[p.lReg2] - 1);
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2]) * 16u) & dataMask;
            loadVfFromMem(addr, p.lReg1, p.lDest);
            return false;
        }
        case FL_SQD:
        {
            if (p.lReg2 != 0u)
                vi[p.lReg2] = static_cast<int16_t>(vi[p.lReg2] - 1);
            const uint32_t addr = (static_cast<uint32_t>(vi[p.lReg2]) * 16u) & dataMask;
            storeVfToMem(addr, p.lReg1, p.lDest);
            return false;
        }
        case FL_ILW:
        case FL_ILWR:
        {
            const int32_t idx = (p.lowerOp == FL_ILW) ? (vi[p.lReg2] + p.imm) : vi[p.lReg2];
            const uint32_t addr = (static_cast<uint32_t>(idx) * 16u) & dataMask;
            if (p.lReg1 != 0u && addr + 16u <= dataSize)
            {
                const uint32_t comp = (p.lDest & 8u) ? 0u : (p.lDest & 4u) ? 1u : (p.lDest & 2u) ? 2u : 3u;
                uint32_t v = 0u;
                std::memcpy(&v, vuData + addr + comp * 4u, 4u);
                vi[p.lReg1] = static_cast<int32_t>(static_cast<int16_t>(v & 0xFFFFu));
            }
            return false;
        }
        case FL_ISW:
        case FL_ISWR:
        {
            const int32_t idx = (p.lowerOp == FL_ISW) ? (vi[p.lReg2] + p.imm) : vi[p.lReg2];
            const uint32_t addr = (static_cast<uint32_t>(idx) * 16u) & dataMask;
            if (addr + 16u <= dataSize)
            {
                const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(vi[p.lReg1] & 0xFFFF));
                for (uint32_t c = 0u; c < 4u; ++c)
                    if ((p.lDest & (8u >> c)) != 0u)
                        std::memcpy(vuData + addr + c * 4u, &val, 4u);
            }
            return false;
        }
        case FL_MOVE:
#if defined(__SSE2__) || defined(_M_X64)
            sseApplyMask(vf[p.lReg1], vf[p.lReg2], p.lDest);
#else
            for (int c = 0; c < 4; ++c)
                if ((p.lDest & (8u >> c)) != 0u)
                    vf[p.lReg1][c] = vf[p.lReg2][c];
#endif
            return false;
        case FL_MR32:
#if defined(__SSE2__) || defined(_M_X64)
            sseApplyMask(vf[p.lReg1], _mm_shuffle_ps(vf[p.lReg2], vf[p.lReg2], _MM_SHUFFLE(0, 3, 2, 1)), p.lDest);
#else
        {
            const float rot[4] = {vf[p.lReg2][1], vf[p.lReg2][2], vf[p.lReg2][3], vf[p.lReg2][0]};
            for (int c = 0; c < 4; ++c)
                if ((p.lDest & (8u >> c)) != 0u)
                    vf[p.lReg1][c] = rot[c];
        }
#endif
            return false;
        case FL_MTIR:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int32_t>(static_cast<int16_t>(getLaneU32(p.lReg2, p.lDest) & 0xFFFFu));
            return false;
        case FL_MFIR:
            if (p.lReg1 != 0u)
            {
                const int32_t v = static_cast<int32_t>(static_cast<int16_t>(vi[p.lReg2] & 0xFFFF));
#if defined(__SSE2__) || defined(_M_X64)
                sseApplyMask(vf[p.lReg1], _mm_castsi128_ps(_mm_set1_epi32(v)), p.lDest);
#else
                const float fv = std::bit_cast<float>(v);
                for (int c = 0; c < 4; ++c)
                    if ((p.lDest & (8u >> c)) != 0u)
                        vf[p.lReg1][c] = fv;
#endif
            }
            return false;
        case FL_IADD:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] + vi[p.lReg3]);
            return false;
        case FL_ISUB:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] - vi[p.lReg3]);
            return false;
        case FL_IADDI:
        case FL_IADDIU:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] + p.imm);
            return false;
        case FL_ISUBIU:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] - p.imm);
            return false;
        case FL_IAND:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] & vi[p.lReg3]);
            return false;
        case FL_IOR:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(vi[p.lReg2] | vi[p.lReg3]);
            return false;
        case FL_DIV:
        {
            const float num = normalizeOperand(getLaneF(p.lReg2, p.lDest & 3u));
            const float den = normalizeOperand(getLaneF(p.lReg1, (p.lDest >> 2) & 3u));
            if (den == 0.0f)
            {
                q = (std::signbit(num) != std::signbit(den))
                        ? -std::numeric_limits<float>::max()
                        : std::numeric_limits<float>::max();
            }
            else
            {
                uint32_t ignored = 0u;
                q = normalizeResult(num / den, ignored);
            }
            return false;
        }
        case FL_SQRT:
        {
            const float v = std::fabs(normalizeOperand(getLaneF(p.lReg1, p.lDest & 3u)));
            uint32_t ignored = 0u;
            q = normalizeResult(std::sqrt(v), ignored);
            return false;
        }
        case FL_RSQRT:
        {
            const float num = normalizeOperand(getLaneF(p.lReg2, p.lDest & 3u));
            const float den = std::fabs(normalizeOperand(getLaneF(p.lReg1, (p.lDest >> 2) & 3u)));
            if (den == 0.0f)
            {
                q = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
            }
            else
            {
                uint32_t ignored = 0u;
                q = normalizeResult(num / std::sqrt(den), ignored);
            }
            return false;
        }
        case FL_WAITQ:
            return false;
        case FL_FCAND:
            vi[1] = ((clip & p.imm24) != 0u) ? 1 : 0;
            return false;
        case FL_FCEQ:
            vi[1] = ((clip & 0xFFFFFFu) == p.imm24) ? 1 : 0;
            return false;
        case FL_FCOR:
            vi[1] = ((clip | p.imm24) == 0xFFFFFFu) ? 1 : 0;
            return false;
        case FL_FCSET:
            clip = p.imm24;
            return false;
        case FL_FMEQ:
        case FL_FMAND:
        case FL_FMOR:
            if (p.lReg1 != 0u)
            {
                // Pass 2 proved a marked producer is visible on every path; m_state.mac is only a fallback.
                uint32_t mac = m_state.mac;
                for (uint32_t k = 0u; k < std::min<uint32_t>(macCount, 8u); ++k)
                {
                    const uint32_t slot = (macCount - 1u - k) & 7u;
                    if (macIssue[slot] + kFmacLatency <= issueCycle)
                    {
                        mac = macValue[slot];
                        break;
                    }
                }
                const uint32_t mask = static_cast<uint16_t>(vi[p.lReg2]);
                vi[p.lReg1] = p.lowerOp == FL_FMEQ ? ((mac & 0xFFFFu) == mask ? 1 : 0)
                                                   : static_cast<int16_t>(p.lowerOp == FL_FMAND ? (mac & mask) : (mac | mask));
            }
            return false;
        case FL_B:
            return true;
        case FL_IBEQ:
            return static_cast<int16_t>(vi[p.lReg1]) == static_cast<int16_t>(vi[p.lReg2]);
        case FL_IBNE:
            return static_cast<int16_t>(vi[p.lReg1]) != static_cast<int16_t>(vi[p.lReg2]);
        case FL_IBLTZ:
            return static_cast<int16_t>(vi[p.lReg1]) < 0;
        case FL_IBGTZ:
            return static_cast<int16_t>(vi[p.lReg1]) > 0;
        case FL_IBLEZ:
            return static_cast<int16_t>(vi[p.lReg1]) <= 0;
        case FL_IBGEZ:
            return static_cast<int16_t>(vi[p.lReg1]) >= 0;
        case FL_XTOP:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(m_state.top & 0x3FFu);
            return false;
        case FL_XITOP:
            if (p.lReg1 != 0u)
                vi[p.lReg1] = static_cast<int16_t>(m_state.itop & 0x3FFu);
            return false;
        case FL_XGKICK:
            xgkickIssued = true;
            xgkickIssueCycle = issueCycle;
            xgkickSourceQword = static_cast<uint16_t>(vi[p.lReg1]);
            return false;
        default:
            return false;
        }
    };

    // A producer runs the reference upper op, so its VF result and MAC flags match the slow path bit for bit.
    auto execProducerUpper = [&](uint32_t pairIdx)
    {
        const uint32_t upper = m_decodedCodeCache[pairIdx].upper;
        const uint8_t dest = DEST(upper);
        const uint8_t fd = FD(upper);
#if defined(__SSE2__) || defined(_M_X64)
        _mm_storeu_ps(m_state.vf[FS(upper)], vf[FS(upper)]);
        _mm_storeu_ps(m_state.vf[FT(upper)], vf[FT(upper)]);
        _mm_storeu_ps(m_state.acc, acc);
#else
        std::memcpy(m_state.vf[FS(upper)], vf[FS(upper)], sizeof(vf[0]));
        std::memcpy(m_state.vf[FT(upper)], vf[FT(upper)], sizeof(vf[0]));
        std::memcpy(m_state.acc, acc, sizeof(acc));
#endif
        m_state.q = q;
        m_state.i = iReg;
#if defined(__SSE2__) || defined(_M_X64)
        // The slow path runs without FTZ/DAZ; a flushed denormal product could change a MADD sum.
        _mm_setcsr(mxcsrGuard.saved);
        execUpper(upper);
        _mm_setcsr((mxcsrGuard.saved & ~0x6000u) | 0xE040u);
#else
        execUpper(upper);
#endif
        uint32_t mac = 0u;
        m_flagPipeline.forEach([&](FlagPipelineEntry &entry) { if (entry.writesMac) mac = entry.mac; });
        m_flagPipeline.clear();
        const bool toAcc = (upper & 0x3Fu) >= 0x3Cu;
#if defined(__SSE2__) || defined(_M_X64)
        if (toAcc)
            sseApplyMask(acc, _mm_loadu_ps(m_state.acc), dest);
        else if (fd != 0u)
            sseApplyMask(vf[fd], _mm_loadu_ps(m_state.vf[fd]), dest);
#else
        float *dst = toAcc ? acc : (fd != 0u ? vf[fd] : nullptr);
        const float *src = toAcc ? m_state.acc : m_state.vf[fd];
        for (int c = 0; dst && c < 4; ++c)
            if ((dest & (8u >> c)) != 0u)
                dst[c] = src[c];
#endif
        m_state.vf[0][0] = 0.0f; m_state.vf[0][1] = 0.0f; m_state.vf[0][2] = 0.0f; m_state.vf[0][3] = 1.0f;
        macIssue[macCount & 7u] = issueCycle;
        macValue[macCount & 7u] = mac;
        ++macCount;
    };

    auto execPair = [&](const FastPair &p, const FastNode &n) -> bool
    {
        if (n.macProducer) [[unlikely]]
        {
            if ((p.flags & kFastFlagShadowLowerFirst) != 0u)
            {
                const bool taken = execFastLower(p);
                execProducerUpper(n.pair);
                return taken;
            }
            execProducerUpper(n.pair);
            return (p.lowerOp != FL_NOP) ? execFastLower(p) : false;
        }
        if ((p.flags & kFastFlagShadowLowerFirst) != 0u)
        {
            const bool taken = execFastLower(p);
            if (p.upperOp != FU_NOP)
                execFastUpper(p);
            return taken;
        }
        if (p.upperOp != FU_NOP)
            execFastUpper(p);
        return (p.lowerOp != FL_NOP) ? execFastLower(p) : false;
    };

    bool programEnded = false;
    uint8_t terminalTailLatency = 0u;

    for (;;)
    {
        const FastNode &n = nodes[nodeIdx];
        const FastPair &p = fast[n.pair];
        const uint64_t nextCycles = cycles + n.cycleStep;
        if (nextCycles > budgetEnd)
        {
            pcIdx = n.pair;
            cycles = budgetEnd;
            break;
        }

        issueCycle = nextCycles - 1u;
        const bool taken = execPair(p, n);
        cycles = nextCycles;

        if ((p.flags & (kFastFlagIsBranch | kFastFlagEBit)) == 0u)
        {
            nodeIdx = n.next;
            continue;
        }

        const FastNode &dn = nodes[n.next];
        const uint32_t delayIdx = dn.pair;
        const FastPair &delay = fast[delayIdx];
        if (cycles + dn.cycleStep > budgetEnd)
        {
            pcIdx = delayIdx;
            if ((p.flags & kFastFlagEBit) != 0u)
            {
                m_state.ebit = true;
                m_state.haltAfterDelaySlot = true;
            }
            else if (taken)
            {
                m_state.branchPending = true;
                m_state.branchTarget = static_cast<uint32_t>(p.imm) * 8u;
                m_state.branchDelay = 1u;
            }
            cycles = budgetEnd;
            break;
        }
        // Like the slow path, the program's last pair leaves the old VI for a branch right after a resume.
        const InstructionUsage &delayLower = m_decodedCodeCache[delayIdx].lowerUsage;
        const uint32_t delayViWrite =
            (p.flags & kFastFlagEBit) != 0u && delayLower.delaysNextBranchRead ? (delayLower.viWrite & ~1u) : 0u;
        const int32_t delayOldVi = delayViWrite != 0u ? vi[std::countr_zero(delayViWrite)] : 0;
        issueCycle = cycles + dn.cycleStep - 1u;
        (void)execPair(delay, dn);
        cycles += dn.cycleStep;

        if ((p.flags & kFastFlagEBit) != 0u)
        {
            if (delayViWrite != 0u)
            {
                m_viBranchBackupReg = static_cast<uint8_t>(std::countr_zero(delayViWrite));
                m_viBranchBackupValue = delayOldVi;
                m_viBranchBackupValid = true;
            }
            pcIdx = (delayIdx + 1u < pairCount) ? (delayIdx + 1u) : 0u;
            terminalTailLatency = dn.tailLatency;
            programEnded = true;
            break;
        }

        nodeIdx = taken ? dn.nextTaken : dn.next;
    }

    // Write back local register state.
#if defined(__SSE2__) || defined(_M_X64)
    for (uint32_t r = 1u; r < 32u; ++r)
        _mm_storeu_ps(m_state.vf[r], vf[r]);
    _mm_storeu_ps(m_state.acc, acc);
#else
    std::memcpy(m_state.vf + 1, vf + 1, sizeof(m_state.vf) - sizeof(m_state.vf[0]));
    std::memcpy(m_state.acc, acc, sizeof(acc));
#endif
    m_state.vf[0][0] = 0.0f; m_state.vf[0][1] = 0.0f; m_state.vf[0][2] = 0.0f; m_state.vf[0][3] = 1.0f;
    std::memcpy(m_state.vi + 1, vi + 1, sizeof(m_state.vi) - sizeof(m_state.vi[0]));
    m_state.vi[0] = 0;
    m_state.q = q;
    m_state.i = iReg;
    m_state.clip = clip;
    m_workingClip = clip;
    m_state.pc = pcIdx * 8u;

    if (programEnded)
    {
        cycles += terminalTailLatency;
        if (xgkickIssued)
        {
            const uint32_t sourceAddress = (xgkickSourceQword * 16u) & dataMask;
            uint32_t copiedBytes = 0u;
            bool validPacket = true;
            for (;;)
            {
                if (copiedBytes > XgkickPipeline::kBufferSize - 16u)
                {
                    reportReservedInstruction(false, 0xFFFFFFFBu);
                    validPacket = false;
                    break;
                }
                const uint32_t tagSrc = (sourceAddress + copiedBytes) & dataMask;
                uint64_t tagLo = 0u;
                std::memcpy(&tagLo, vuData + tagSrc, sizeof(tagLo));
                const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
                uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                uint64_t tagBytes = 16u;
                if (format == 0u)
                    tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
                else if (format == 1u)
                    tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
                else if (format == 2u || format == 3u)
                    tagBytes += static_cast<uint64_t>(nloop) * 16u;
                else
                {
                    reportReservedInstruction(false, 0xFFFFFFF8u);
                    validPacket = false;
                    break;
                }
                if (tagBytes > XgkickPipeline::kBufferSize - copiedBytes)
                {
                    reportReservedInstruction(false, 0xFFFFFFFBu);
                    validPacket = false;
                    break;
                }
                const uint32_t copyLen = static_cast<uint32_t>(tagBytes);
                uint32_t remaining = copyLen;
                uint32_t srcPos = (sourceAddress + copiedBytes) & dataMask;
                uint8_t *dstPtr = m_xgkick.packet.data() + copiedBytes;
                while (remaining > 0u)
                {
                    const uint32_t chunk = std::min(remaining, dataSize - srcPos);
                    std::memcpy(dstPtr, vuData + srcPos, chunk);
                    dstPtr += chunk;
                    srcPos = (srcPos + chunk) & dataMask;
                    remaining -= chunk;
                }
                copiedBytes += copyLen;
                const bool eop = ((tagLo >> 15) & 1u) != 0u;
                if (eop)
                    break;
            }
            if (validPacket)
            {
                const uint64_t totalQwords = copiedBytes / 16u;
                const uint64_t xgkickDoneCycle = xgkickIssueCycle + (totalQwords > 0u ? (2u * totalQwords - 1u) : 0u);
                if (xgkickDoneCycle > cycles)
                    cycles = xgkickDoneCycle;
                if (m_activeMemory)
                    m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), copiedBytes);
                else if (m_activeGs)
                    m_activeGs->processGIFPacket(m_xgkick.packet.data(), copiedBytes);
            }
        }
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    else if (xgkickIssued)
    {
        m_cycle = xgkickIssueCycle;
        startXgkick(xgkickSourceQword);
        while (m_cycle < cycles)
        {
            ++m_cycle;
            progressXgkick();
        }
    }

    m_cycle = cycles;
    m_state.cycles = m_cycle;
}
