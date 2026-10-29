#include "ComboStep.h"

#include <algorithm>

namespace GameLogic::Combat
{
    std::optional<ComboStep> ParseComboStep(std::span<const std::uint8_t> packet)
    {
        if (packet.size() < kComboStepPacketSize || packet[0] != 0xC1 || packet[1] < kComboStepPacketSize
            || packet[2] != kComboStepPacketCode || packet[3] != kComboStepPacketSubCode)
        {
            return std::nullopt;
        }

        ComboStep step;
        step.attackerKey = (packet[4] << 8) | packet[5];
        step.targetKey = ((packet[6] << 8) | packet[7]) & 0x7FFF;
        step.step = packet[8];
        if (step.step < 2 || step.step > kLastComboStep)
        {
            return std::nullopt;
        }

        return step;
    }

    std::array<float, 3> ComboStepTint(int step)
    {
        static constexpr std::array<std::array<float, 3>, kLastComboStep> kTints = {{
            {1.00f, 1.00f, 1.00f},
            {0.50f, 0.70f, 1.35f},
            {0.25f, 0.45f, 1.70f},
            {0.08f, 0.25f, 2.10f},
        }};
        return kTints[static_cast<std::size_t>(std::clamp(step, 1, kLastComboStep) - 1)];
    }
}
