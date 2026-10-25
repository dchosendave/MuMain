#pragma once

#include <algorithm>
#include <cstdint>

namespace Engine::Object
{
    // Classic swing rhythm: at high attack speed the hero's attack swings, and the decision to
    // start the next cast, update 30 times per second as in the 30 FPS clients reset servers are
    // played with, while everything else keeps rendering at the full frame rate.
    //
    // At 2,000+ attack speed a swing lasts about two 25 FPS reference ticks. A 30 FPS client
    // therefore only ever shows the hit pose ("cast once, attached"), and re-picks the target and
    // the next cast 30 times per second. At 180 FPS the same swing was replayed over 13 frames
    // and the cast was re-decided 180 times per second, which landed 2-3 times fewer combos in the
    // same time (measured on recordings of both). Slow swings keep animating at the full frame
    // rate, so low attack speeds stay smooth.
    constexpr int kClassicSwingRateHz = 30;
    constexpr int kClassicSwingMinimumAttackSpeed = 1000;
    constexpr int kClassicSwingMaximumStepsPerFrame = 2;

    // Toggled with the chat command "$classicswing on|off".
    inline bool g_ClassicSwingEnabled = true;

    inline bool IsClassicSwingActive(int attackSpeed)
    {
        return g_ClassicSwingEnabled && attackSpeed >= kClassicSwingMinimumAttackSpeed;
    }

    // The 30 Hz tick a millisecond timestamp falls into.
    inline std::uint64_t ClassicSwingTick(std::uint32_t nowMs)
    {
        return static_cast<std::uint64_t>(nowMs) * kClassicSwingRateHz / 1000;
    }

    // Counts the 30 Hz steps between two calls. Calling it more than once per frame is harmless:
    // a second call in the same tick returns 0.
    class ClassicSwingClock
    {
    public:
        int Steps(std::uint32_t nowMs)
        {
            const auto tick = ClassicSwingTick(nowMs);
            if (!m_started || tick < m_lastTick)
            {
                m_started = true;
                m_lastTick = tick;
                return 0;
            }

            const auto steps = static_cast<int>(std::min<std::uint64_t>(tick - m_lastTick, kClassicSwingMaximumStepsPerFrame));
            m_lastTick = tick;
            return steps;
        }

    private:
        bool m_started = false;
        std::uint64_t m_lastTick = 0;
    };

    // The PlaySpeed which, multiplied with this frame's animation factor, advances the swing as
    // far as `steps` frames of a 30 FPS client do, and not at all between the steps.
    inline float ClassicSwingPlaySpeed(float playSpeed, int steps, float frameAnimationFactor)
    {
        if (steps <= 0 || frameAnimationFactor <= 0.f)
        {
            return 0.f;
        }

        constexpr float kThirtyFpsAnimationFactor = 25.f / kClassicSwingRateHz;
        return playSpeed * kThirtyFpsAnimationFactor * static_cast<float>(steps) / frameAnimationFactor;
    }

    inline ClassicSwingClock g_HeroSwingAnimationClock;
    inline ClassicSwingClock g_HeroCastDecisionClock;
}
