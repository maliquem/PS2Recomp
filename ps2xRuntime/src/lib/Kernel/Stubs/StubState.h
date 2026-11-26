#pragma once

namespace ps2x::state
{
    class Archive;
}

// Save state of the HLE stubs. Their globals are file-local, so each stub source defines its own
// function at its end; Kernel/SaveState.cpp calls them in this order.
namespace ps2_stubs
{
    void audioStubStateIo(ps2x::state::Archive &ar);
    void cdStubStateIo(ps2x::state::Archive &ar);
    void dmaStubStateIo(ps2x::state::Archive &ar);
    void gsStubStateIo(ps2x::state::Archive &ar);
    void libcStubStateIo(ps2x::state::Archive &ar);
    void mcStubStateIo(ps2x::state::Archive &ar);
    void mpegStubStateIo(ps2x::state::Archive &ar);
    void padStubStateIo(ps2x::state::Archive &ar);
    void sifStubStateIo(ps2x::state::Archive &ar);

    // Why a save has to wait (a file or a movie in flight), or nullptr.
    const char *libcStubBusy();
    const char *mcStubBusy();
    const char *mpegStubBusy();
}
