#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace GameLogic::Combat
{
    // A follow-up combo of OpenMU's combo chain, as the server sends it with the custom packet
    // C1 09 FB 02 (ShowComboStepPlugIn): attacker and target key (big endian, like the skill animation)
    // and the step, 2 to 4. The first combo stays the usual skill animation.
    constexpr std::uint8_t kComboStepPacketCode = 0xFB;
    constexpr std::uint8_t kComboStepPacketSubCode = 0x02;
    constexpr std::size_t kComboStepPacketSize = 9;
    constexpr int kLastComboStep = 4;

    struct ComboStep
    {
        int attackerKey = 0;
        int targetKey = 0;
        int step = 0;
    };

    [[nodiscard]] std::optional<ComboStep> ParseComboStep(std::span<const std::uint8_t> packet);

    // How a step tints the combo circle, multiplied into the attacker's light: each step a deeper,
    // more saturated blue than the one before (less red and green, more blue). Step 1 keeps the
    // classic look.
    [[nodiscard]] std::array<float, 3> ComboStepTint(int step);
}
