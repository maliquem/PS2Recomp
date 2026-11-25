// Shared PS2 GS VRAM addressing, CLUT, texture sampling, and pixel write pipeline.
// Expects binding 0 = vram[], binding 1 = clut[] (512 uint16 packed in 256 uint32).

const uint PSM_CT32  = 0u;
const uint PSM_CT24  = 1u;
const uint PSM_CT16  = 2u;
const uint PSM_CT16S = 10u;
const uint PSM_T8    = 19u;
const uint PSM_T4    = 20u;
const uint PSM_T8H   = 27u;
const uint PSM_T4HL  = 36u;
const uint PSM_T4HH  = 44u;
const uint PSM_Z32   = 48u;
const uint PSM_Z24   = 49u;
const uint PSM_Z16   = 50u;
const uint PSM_Z16S  = 58u;

uint blockC32(uint bx, uint by)
{
    return (bx & 1u) | ((by & 1u) << 1u) | ((bx & 2u) << 1u) |
           ((by & 2u) << 2u) | ((bx & 4u) << 2u);
}

uint blockC16(uint bx, uint by)
{
    return (by & 1u) | ((bx & 1u) << 1u) | ((by & 2u) << 1u) |
           ((bx & 2u) << 2u) | ((by & 4u) << 2u);
}

uint blockC16S(uint bx, uint by)
{
    return (by & 1u) | ((bx & 1u) << 1u) | (by & 4u) |
           ((by & 2u) << 2u) | ((bx & 2u) << 3u);
}

uint addrWord32(uint bp, uint bw, uint x, uint y, uint xorMask)
{
    uint page = (bp >> 5u) + (y >> 5u) * bw + (x >> 6u);
    uint blk = (bp & 31u) + (blockC32((x >> 3u) & 7u, (y >> 3u) & 3u) ^ xorMask);
    uint col = (x & 1u) | ((y & 1u) << 1u) | ((x & 6u) << 1u) | ((y & 6u) << 3u);
    return ((page * 32u + blk) * 64u + col) & 0xFFFFFu;
}

uint addrHalf16(uint bp, uint bw, uint x, uint y, uint psm)
{
    uint page = (bp >> 5u) + (y >> 6u) * bw + (x >> 6u);
    uint bx = (x >> 4u) & 3u;
    uint by = (y >> 3u) & 7u;
    uint blk;
    if (psm == PSM_CT16)
        blk = blockC16(bx, by);
    else if (psm == PSM_CT16S)
        blk = blockC16S(bx, by);
    else if (psm == PSM_Z16)
        blk = blockC16(bx, by) ^ 24u;
    else
        blk = blockC16S(bx, by) ^ 24u;
    uint col = ((x & 8u) >> 3u) | ((x & 1u) << 1u) | ((y & 1u) << 2u) |
               ((x & 6u) << 2u) | ((y & 6u) << 4u);
    return ((page * 32u + (bp & 31u) + blk) * 128u + col) & 0x1FFFFFu;
}

uint addrByteP8(uint bp, uint bw, uint x, uint y)
{
    uint page = (bp >> 5u) + (y >> 6u) * (bw >> 1u) + (x >> 7u);
    uint blk = (bp & 31u) + blockC32((x >> 4u) & 7u, (y >> 4u) & 3u);
    uint y0 = y & 1u;
    uint y1 = (y >> 1u) & 1u;
    uint y2 = (y >> 2u) & 1u;
    uint y3 = (y >> 3u) & 1u;
    uint fx2 = ((x >> 2u) & 1u) ^ y1 ^ y2;
    uint col = y1 | (((x >> 3u) & 1u) << 1u) | ((x & 1u) << 2u) | (y0 << 3u) |
               (((x >> 1u) & 1u) << 4u) | (fx2 << 5u) | (y2 << 6u) | (y3 << 7u);
    return ((page * 32u + blk) * 256u + col) & 0x3FFFFFu;
}

uint addrNibbleP4(uint bp, uint bw, uint x, uint y)
{
    uint page = (bp >> 5u) + (y >> 7u) * (bw >> 1u) + (x >> 7u);
    uint blk = (bp & 31u) + blockC16((x >> 5u) & 3u, (y >> 4u) & 7u);
    uint y0 = y & 1u;
    uint y1 = (y >> 1u) & 1u;
    uint y2 = (y >> 2u) & 1u;
    uint y3 = (y >> 3u) & 1u;
    uint fx2 = ((x >> 2u) & 1u) ^ y1 ^ y2;
    uint col = y1 | ((x & 24u) >> 2u) | ((x & 1u) << 3u) | (y0 << 4u) |
               ((x & 2u) << 4u) | (fx2 << 6u) | (y2 << 7u) | (y3 << 8u);
    return ((page * 32u + blk) * 512u + col) & 0x7FFFFFu;
}

uint readVram(uint psm, uint bp, uint bw, uint x, uint y)
{
    if (psm == PSM_CT32)
        return vram[addrWord32(bp, bw, x, y, 0u)];
    if (psm == PSM_CT24)
        return vram[addrWord32(bp, bw, x, y, 0u)] & 0x00FFFFFFu;
    if (psm == PSM_Z32)
        return vram[addrWord32(bp, bw, x, y, 24u)];
    if (psm == PSM_Z24)
        return vram[addrWord32(bp, bw, x, y, 24u)] & 0x00FFFFFFu;
    if (psm == PSM_CT16 || psm == PSM_CT16S || psm == PSM_Z16 || psm == PSM_Z16S)
    {
        uint h = addrHalf16(bp, bw, x, y, psm);
        return (vram[h >> 1u] >> ((h & 1u) * 16u)) & 0xFFFFu;
    }
    if (psm == PSM_T8)
    {
        uint b = addrByteP8(bp, bw, x, y);
        return (vram[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu;
    }
    if (psm == PSM_T8H)
        return (vram[addrWord32(bp, bw, x, y, 0u)] >> 24u) & 0xFFu;
    if (psm == PSM_T4)
    {
        uint n = addrNibbleP4(bp, bw, x, y);
        return (vram[n >> 3u] >> ((n & 7u) * 4u)) & 0x0Fu;
    }
    if (psm == PSM_T4HL)
        return (vram[addrWord32(bp, bw, x, y, 0u)] >> 24u) & 0x0Fu;
    if (psm == PSM_T4HH)
        return (vram[addrWord32(bp, bw, x, y, 0u)] >> 28u) & 0x0Fu;
    return 0xFFFF00FFu;
}

void atomicWriteMasked(uint wordIdx, uint mask, uint shiftedVal)
{
    uint cur = atomicOr(vram[wordIdx], 0u);
    for (;;)
    {
        uint next = (cur & ~mask) | (shiftedVal & mask);
        if (next == cur)
            return;
        uint prev = atomicCompSwap(vram[wordIdx], cur, next);
        if (prev == cur)
            return;
        cur = prev;
    }
}

void writeVram(uint psm, uint bp, uint bw, uint x, uint y, uint value)
{
    if (psm == PSM_CT32)
    {
        vram[addrWord32(bp, bw, x, y, 0u)] = value;
        return;
    }
    if (psm == PSM_Z32)
    {
        vram[addrWord32(bp, bw, x, y, 24u)] = value;
        return;
    }
    if (psm == PSM_CT24)
    {
        atomicWriteMasked(addrWord32(bp, bw, x, y, 0u), 0x00FFFFFFu, value);
        return;
    }
    if (psm == PSM_Z24)
    {
        atomicWriteMasked(addrWord32(bp, bw, x, y, 24u), 0x00FFFFFFu, value);
        return;
    }
    if (psm == PSM_CT16 || psm == PSM_CT16S || psm == PSM_Z16 || psm == PSM_Z16S)
    {
        uint h = addrHalf16(bp, bw, x, y, psm);
        uint shift = (h & 1u) * 16u;
        atomicWriteMasked(h >> 1u, 0xFFFFu << shift, (value & 0xFFFFu) << shift);
        return;
    }
    if (psm == PSM_T8)
    {
        uint b = addrByteP8(bp, bw, x, y);
        uint shift = (b & 3u) * 8u;
        atomicWriteMasked(b >> 2u, 0xFFu << shift, (value & 0xFFu) << shift);
        return;
    }
    if (psm == PSM_T8H)
    {
        atomicWriteMasked(addrWord32(bp, bw, x, y, 0u), 0xFF000000u, (value & 0xFFu) << 24u);
        return;
    }
    if (psm == PSM_T4)
    {
        uint n = addrNibbleP4(bp, bw, x, y);
        uint shift = (n & 7u) * 4u;
        atomicWriteMasked(n >> 3u, 0x0Fu << shift, (value & 0x0Fu) << shift);
        return;
    }
    if (psm == PSM_T4HL)
    {
        atomicWriteMasked(addrWord32(bp, bw, x, y, 0u), 0x0F000000u, (value & 0x0Fu) << 24u);
        return;
    }
    if (psm == PSM_T4HH)
    {
        atomicWriteMasked(addrWord32(bp, bw, x, y, 0u), 0xF0000000u, (value & 0x0Fu) << 28u);
        return;
    }
}

uint readClutU16(uint idx)
{
    uint w = clut[idx >> 1u];
    return (w >> ((idx & 1u) * 16u)) & 0xFFFFu;
}

uint rgba5551To8888(uint c)
{
    uint r = ((c >> 0u) & 0x1Fu) << 3u;
    uint g = ((c >> 5u) & 0x1Fu) << 3u;
    uint b = ((c >> 10u) & 0x1Fu) << 3u;
    uint a = ((c >> 15u) & 0x01u) << 7u;
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

uint rgba8888To5551(uint c)
{
    uint r = ((c >> 0u) & 0xFFu) >> 3u;
    uint g = ((c >> 8u) & 0xFFu) >> 3u;
    uint b = ((c >> 16u) & 0xFFu) >> 3u;
    uint a = ((c >> 24u) & 0xFFu) >> 7u;
    return r | (g << 5u) | (b << 10u) | (a << 15u);
}

uint applyTexa(uint texaPacked, uint psm, uint texel)
{
    if (psm == PSM_CT32)
        return texel;
    uint ta0 = texaPacked & 0xFFu;
    uint ta1 = (texaPacked >> 8u) & 0xFFu;
    bool aem = ((texaPacked >> 16u) & 1u) != 0u;
    bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
    uint a = (texel >> 24u) & 0xFFu;
    if (psm == PSM_CT24)
        a = (aem && rgbZero) ? 0u : ta0;
    else if (psm == PSM_CT16 || psm == PSM_CT16S)
        a = ((a & 0x80u) != 0u) ? ta1 : ((aem && rgbZero) ? 0u : ta0);
    return (texel & 0x00FFFFFFu) | (a << 24u);
}

uint lookupClut(uint texaPacked, uint idx, uint cpsm, uint csm, uint csa, uint srcPsm)
{
    bool sixteen = (cpsm == PSM_CT16 || cpsm == PSM_CT16S);
    bool fourBit = (srcPsm == PSM_T4 || srcPsm == PSM_T4HL || srcPsm == PSM_T4HH);
    bool eightBit = (srcPsm == PSM_T8 || srcPsm == PSM_T8H);
    uint csaMask = sixteen ? 0x1Fu : 0x0Fu;
    uint clutBase = (csa & csaMask) << 4u;
    uint srcIdx = fourBit ? (idx & 0x0Fu) : (idx & 0xFFu);
    uint ci = (clutBase + srcIdx) & (sixteen ? 0x1FFu : 0x0FFu);
    if (!sixteen && csm == 0u && eightBit)
    {
        uint blk = min((srcIdx & 0xF0u) + clutBase, 240u);
        ci = blk + (srcIdx & 0x0Fu);
    }
    if (cpsm == PSM_CT32)
    {
        uint raw = readClutU16(ci) | (readClutU16(ci + 256u) << 16u);
        return applyTexa(texaPacked, cpsm, raw);
    }
    if (cpsm == PSM_CT24)
    {
        uint raw = readClutU16(ci) | (readClutU16(ci + 256u) << 16u);
        return applyTexa(texaPacked, cpsm, raw & 0x00FFFFFFu);
    }
    if (cpsm == PSM_CT16 || cpsm == PSM_CT16S)
        return applyTexa(texaPacked, cpsm, rgba5551To8888(readClutU16(ci)));
    return 0xFFFF00FFu;
}

int wrapTexCoord(int coord, int texSize, uint mode, uint rMin, uint rMax)
{
    uint m = mode & 3u;
    if (m == 0u)
        return int(uint(coord) & uint(texSize - 1));
    if (m == 1u)
        return clamp(coord, 0, texSize - 1);
    if (m == 2u)
        return min(max(coord, int(rMin)), int(rMax));
    return int((uint(coord) & rMin) | rMax);
}

uint sampleTexelPoint(uint tbp0, uint tbw, uint tpsm, int texW, int texH,
                      uint clampLo, uint clampHi, uint texaPacked,
                      uint cpsm, uint csm, uint csa, int u, int v)
{
    uint wrapU = clampLo & 3u;
    uint wrapV = (clampLo >> 2u) & 3u;
    uint minU = (clampLo >> 4u) & 0x3FFu;
    uint maxU = (clampLo >> 14u) & 0x3FFu;
    uint minV = ((clampLo >> 24u) & 0xFFu) | ((clampHi & 3u) << 8u);
    uint maxV = (clampHi >> 2u) & 0x3FFu;
    int su = wrapTexCoord(u, texW, wrapU, minU, maxU);
    int sv = wrapTexCoord(v, texH, wrapV, minV, maxV);
    uint raw = readVram(tpsm, tbp0, max(tbw, 1u), uint(su), uint(sv));
    if (tpsm == PSM_CT32 || tpsm == PSM_Z32 || tpsm == PSM_CT24 || tpsm == PSM_Z24)
        return applyTexa(texaPacked, tpsm, raw);
    if (tpsm == PSM_CT16 || tpsm == PSM_CT16S || tpsm == PSM_Z16 || tpsm == PSM_Z16S)
        return applyTexa(texaPacked, tpsm, rgba5551To8888(raw));
    if (tpsm == PSM_T8 || tpsm == PSM_T8H || tpsm == PSM_T4 || tpsm == PSM_T4HL || tpsm == PSM_T4HH)
        return lookupClut(texaPacked, raw, cpsm, csm, csa, tpsm);
    return 0xFFFF00FFu;
}

ivec4 unpackRgba(uint c)
{
    return ivec4(uvec4(c, c >> 8u, c >> 16u, c >> 24u) & 255u);
}

uint packRgba(ivec4 c)
{
    uvec4 u = uvec4(clamp(c, 0, 255));
    return u.r | (u.g << 8u) | (u.b << 16u) | (u.a << 24u);
}

ivec4 combineTex(uint tfx, uint tcc, ivec4 vc, ivec4 tc)
{
    bool hasAlpha = (tcc != 0u);
    ivec4 outC = ivec4(tc.rgb, hasAlpha ? tc.a : vc.a);
    if (tfx == 0u)
    {
        outC.rgb = clamp((tc.rgb * vc.rgb) >> 7, 0, 255);
        outC.a = hasAlpha ? clamp((tc.a * vc.a) >> 7, 0, 255) : vc.a;
    }
    else if (tfx == 2u)
    {
        outC.rgb = clamp(((tc.rgb * vc.rgb) >> 7) + vc.a, 0, 255);
        outC.a = hasAlpha ? clamp(tc.a + vc.a, 0, 255) : vc.a;
    }
    else if (tfx == 3u)
    {
        outC.rgb = clamp(((tc.rgb * vc.rgb) >> 7) + vc.a, 0, 255);
        outC.a = hasAlpha ? tc.a : vc.a;
    }
    return outC;
}

bool passesAlphaTest(uint testLo, int alpha)
{
    if ((testLo & 1u) == 0u)
        return true;
    uint atst = (testLo >> 1u) & 7u;
    int aref = int((testLo >> 4u) & 0xFFu);
    if (atst == 0u) return false;
    if (atst == 1u) return true;
    if (atst == 2u) return alpha < aref;
    if (atst == 3u) return alpha <= aref;
    if (atst == 4u) return alpha == aref;
    if (atst == 5u) return alpha >= aref;
    if (atst == 6u) return alpha > aref;
    return alpha != aref;
}

int pickRgbSel(uint sel, int cs, int cd)
{
    if (sel == 0u) return cs;
    if (sel == 1u) return cd;
    return 0;
}

void writeGsPixel(uint x, uint y, uint zVal, ivec4 color, uint fog,
                  uint fbpBlock, uint fbw, uint fpsm, uint fbmsk,
                  uint zbpBlock, uint zpsm, bool zmask,
                  uint testLo, uint alphaLo, uint alphaFix,
                  bool abe, bool pabe, bool fba, bool fge, ivec3 fogColor)
{
    if (fge)
    {
        uint invFog = 255u - fog;
        color.r = int(((fog * uint(color.r)) >> 8u) + ((invFog * uint(fogColor.r)) >> 8u));
        color.g = int(((fog * uint(color.g)) >> 8u) + ((invFog * uint(fogColor.g)) >> 8u));
        color.b = int(((fog * uint(color.b)) >> 8u) + ((invFog * uint(fogColor.b)) >> 8u));
    }

    bool writeRgb = true;
    bool writeAlpha = true;
    bool writeDepth = true;
    if (!passesAlphaTest(testLo, color.a))
    {
        uint afail = (testLo >> 12u) & 3u;
        if (afail == 0u) return;
        if (afail == 1u) { writeDepth = false; }
        else if (afail == 2u) { writeRgb = false; writeAlpha = false; }
        else
        {
            writeDepth = false;
            if (fpsm == PSM_CT32)
                writeAlpha = false;
        }
    }

    bool writesFb = writeRgb || writeAlpha;
    bool preserveDstAlpha = writeRgb && !writeAlpha && fpsm == PSM_CT32;
    bool date = ((testLo >> 14u) & 1u) != 0u;
    bool dateNeedsRead = date && (fpsm == PSM_CT32 || fpsm == PSM_CT16 || fpsm == PSM_CT16S);
    bool frmw = dateNeedsRead || (writesFb && (fbmsk != 0u || abe || preserveDstAlpha));

    uint rawFb = 0u;
    uint fbRgba = 0u;
    if (frmw)
    {
        rawFb = readVram(fpsm, fbpBlock, fbw, x, y);
        fbRgba = rawFb;
        if (fpsm == PSM_CT16 || fpsm == PSM_CT16S)
            fbRgba = rgba5551To8888(fbRgba);
        else if (fpsm == PSM_CT24)
            fbRgba |= 0x80000000u;
    }

    if (dateNeedsRead)
    {
        bool datm = ((testLo >> 15u) & 1u) != 0u;
        uint bit = (fpsm == PSM_CT32) ? ((rawFb >> 31u) & 1u) : ((rawFb >> 15u) & 1u);
        if ((bit != 0u) != datm)
            return;
    }

    uint ztst = (testLo >> 17u) & 3u;
    if (ztst == 0u)
        return;
    if (ztst >= 2u)
    {
        uint storedZ = readVram(zpsm, zbpBlock, fbw, x, y);
        bool zpass = (ztst == 2u) ? (zVal >= storedZ) : (zVal > storedZ);
        if (!zpass)
            return;
    }

    if (writesFb)
    {
        ivec3 outRgb = color.rgb;
        int outA = color.a;
        if (abe && !(pabe && (outA & 0x80) == 0))
        {
            ivec4 dstC = unpackRgba(fbRgba);
            uint asel = alphaLo & 3u;
            uint bsel = (alphaLo >> 2u) & 3u;
            uint csel = (alphaLo >> 4u) & 3u;
            uint dsel = (alphaLo >> 6u) & 3u;
            int cAlpha = (csel == 0u) ? outA : ((csel == 1u) ? dstC.a : int(alphaFix));
            outRgb.r = clamp((((pickRgbSel(asel, outRgb.r, dstC.r) - pickRgbSel(bsel, outRgb.r, dstC.r)) * cAlpha) >> 7) + pickRgbSel(dsel, outRgb.r, dstC.r), 0, 255);
            outRgb.g = clamp((((pickRgbSel(asel, outRgb.g, dstC.g) - pickRgbSel(bsel, outRgb.g, dstC.g)) * cAlpha) >> 7) + pickRgbSel(dsel, outRgb.g, dstC.g), 0, 255);
            outRgb.b = clamp((((pickRgbSel(asel, outRgb.b, dstC.b) - pickRgbSel(bsel, outRgb.b, dstC.b)) * cAlpha) >> 7) + pickRgbSel(dsel, outRgb.b, dstC.b), 0, 255);
        }
        if (writeAlpha && fba && fpsm != PSM_CT24)
            outA |= 0x80;
        uint px = packRgba(ivec4(outRgb, outA));
        if (fbmsk != 0u)
            px = (px & ~fbmsk) | (fbRgba & fbmsk);
        if (preserveDstAlpha)
            px = (px & 0x00FFFFFFu) | (fbRgba & 0xFF000000u);
        if (fpsm == PSM_CT16 || fpsm == PSM_CT16S)
            px = rgba8888To5551(px);
        writeVram(fpsm, fbpBlock, fbw, x, y, px);
    }

    if (writeDepth && !zmask)
        writeVram(zpsm, zbpBlock, fbw, x, y, zVal);
}
