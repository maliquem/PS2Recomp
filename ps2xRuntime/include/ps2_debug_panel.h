#ifndef PS2_DEBUG_PANEL_H
#define PS2_DEBUG_PANEL_H

class PS2Runtime;

class PS2DebugPanel
{
public:
    void initialize();
    void shutdown();
    void draw(PS2Runtime &runtime);

    bool isVisible() const { return m_visible; }
    void setVisible(bool visible) { m_visible = visible; }
    void toggleVisible() { m_visible = !m_visible; }

private:
    bool m_initialized = false;
    bool m_visible = true;
    bool m_showRegisters = true;
    unsigned int m_memoryAddress = 0x00100000u;
    unsigned int m_memoryBytes = 0x100u;
    // F3 counter: presents per second (FPS) and EE VBlanks per second (VPS).
    bool m_showRate = false;
    unsigned int m_rateFrames = 0;
    unsigned long long m_rateTick = 0;
    unsigned long long m_rateStartNs = 0;
    float m_fps = 0.0f, m_vps = 0.0f;
};

#endif // PS2_DEBUG_PANEL_H
