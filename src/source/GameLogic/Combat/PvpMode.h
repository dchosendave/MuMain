#pragma once

#include <cstdint>

namespace GameLogic::Combat
{
    // PvP mode: while it's on, attacking a player works as if left Ctrl were held, except on party and
    // guild members, which still need the real Ctrl. A double tap of left Ctrl turns it on or off.
    [[nodiscard]] bool IsPvpModeOn();
    void SetPvpMode(bool on);

    // Recognizes a double tap of left Ctrl: two clean taps (pressed and released quickly, with no
    // other key or mouse button in between), the second released soon after the first. Holding Ctrl
    // for Ctrl + number or Ctrl + click never counts.
    class CtrlDoubleTap
    {
    public:
        static constexpr std::uint64_t kMaxTapMs = 250;
        static constexpr std::uint64_t kMaxGapMs = 400;

        void OnCtrlDown(std::uint64_t nowMs);

        // Returns true when this release completes a double tap.
        [[nodiscard]] bool OnCtrlUp(std::uint64_t nowMs);

        // Any other key or mouse button: a Ctrl held for it isn't a tap, and it breaks a pending one.
        void OnOtherInput();

    private:
        bool m_down = false;
        bool m_dirty = false;
        std::uint64_t m_downAt = 0;
        std::uint64_t m_lastTapAt = 0;
        bool m_hasLastTap = false;
    };
}
