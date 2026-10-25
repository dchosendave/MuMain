#pragma once

#include <array>
#include <cstdint>

namespace GameLogic::Combat
{
    // Skill hotkeys pressed while the hero is still swinging, cast in the order they were pressed.
    // Without it a hotkey only replaces the current skill, so the skill which happens to be
    // selected when the swing ends gets cast: a fast 1-2-3 combo macro then casts e.g. 1-3 and
    // breaks the combo. Repeated presses of the same skill don't stack, a full queue ignores
    // further presses (dropping the oldest would skip a step), and presses which weren't cast
    // within kMaximumAgeMs expire, so a forgotten press never fires later.
    class SkillInputQueue
    {
    public:
        static constexpr int kCapacity = 3;
        static constexpr std::uint32_t kMaximumAgeMs = 1000;

        void Push(int skillIndex, std::uint32_t nowMs)
        {
            if (m_count > 0 && m_entries[(m_first + m_count - 1) % kCapacity].SkillIndex == skillIndex)
            {
                return;
            }

            if (m_count == kCapacity)
            {
                return;
            }

            m_entries[(m_first + m_count) % kCapacity] = { skillIndex, nowMs };
            ++m_count;
        }

        // The next skill to cast, or -1 when there is none.
        int Peek(std::uint32_t nowMs)
        {
            while (m_count > 0 && nowMs - m_entries[m_first].PressedAtMs > kMaximumAgeMs)
            {
                Pop();
            }

            return m_count > 0 ? m_entries[m_first].SkillIndex : -1;
        }

        void Pop()
        {
            if (m_count == 0)
            {
                return;
            }

            m_first = (m_first + 1) % kCapacity;
            --m_count;
        }

        void Clear()
        {
            m_first = 0;
            m_count = 0;
        }

        bool Empty() const { return m_count == 0; }

    private:
        struct Entry
        {
            int SkillIndex = -1;
            std::uint32_t PressedAtMs = 0;
        };

        std::array<Entry, kCapacity> m_entries{};
        int m_first = 0;
        int m_count = 0;
    };

    // The hero's queue. Filled by the skill hotkeys, used by Attack() in ZzzInterface.cpp.
    inline SkillInputQueue g_HeroSkillInputQueue;
}
