#include "PvpMode.h"

namespace GameLogic::Combat
{
    namespace
    {
        bool g_pvpMode = false;
    }

    bool IsPvpModeOn()
    {
        return g_pvpMode;
    }

    void SetPvpMode(bool on)
    {
        g_pvpMode = on;
    }

    void CtrlDoubleTap::OnCtrlDown(std::uint64_t nowMs)
    {
        if (m_down)
        {
            return; // key repeat
        }

        m_down = true;
        m_dirty = false;
        m_downAt = nowMs;
    }

    bool CtrlDoubleTap::OnCtrlUp(std::uint64_t nowMs)
    {
        if (!m_down)
        {
            return false;
        }

        m_down = false;
        if (m_dirty || nowMs - m_downAt > kMaxTapMs)
        {
            m_hasLastTap = false;
            return false;
        }

        if (m_hasLastTap && nowMs - m_lastTapAt <= kMaxGapMs)
        {
            m_hasLastTap = false;
            return true;
        }

        m_hasLastTap = true;
        m_lastTapAt = nowMs;
        return false;
    }

    void CtrlDoubleTap::OnOtherInput()
    {
        m_dirty = m_down;
        m_hasLastTap = false;
    }
}
