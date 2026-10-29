#include <doctest.h>

#include "GameLogic/Combat/ComboStep.h"

#include <array>
#include <cstdint>

using GameLogic::Combat::ComboStepTint;
using GameLogic::Combat::ParseComboStep;

TEST_CASE("The combo step packet is the one OpenMU's ShowComboStepPlugIn writes")
{
    // OpenMU's ComboStepPacketTest checks the same bytes.
    const std::array<std::uint8_t, 9> packet = {0xC1, 0x09, 0xFB, 0x02, 0x12, 0x34, 0x0A, 0x0B, 0x03};

    const auto step = ParseComboStep(packet);

    REQUIRE(step.has_value());
    CHECK(step->attackerKey == 0x1234);
    CHECK(step->targetKey == 0x0A0B);
    CHECK(step->step == 3);
}

TEST_CASE("Other packets and steps out of range are no combo step")
{
    std::array<std::uint8_t, 9> packet = {0xC1, 0x09, 0xFB, 0x02, 0, 1, 0, 2, 2};
    CHECK(ParseComboStep(packet).has_value());

    packet[3] = 0x01;  // the Castle Siege scoreboard
    CHECK_FALSE(ParseComboStep(packet).has_value());
    packet[3] = 0x02;

    packet[8] = 1;
    CHECK_FALSE(ParseComboStep(packet).has_value());
    packet[8] = 5;
    CHECK_FALSE(ParseComboStep(packet).has_value());
    CHECK_FALSE(ParseComboStep(std::span<const std::uint8_t>(packet).first(8)).has_value());
}

TEST_CASE("Each combo step is a deeper blue than the one before")
{
    CHECK(ComboStepTint(1) == std::array<float, 3>{1.f, 1.f, 1.f});
    for (int step = 2; step <= GameLogic::Combat::kLastComboStep; ++step)
    {
        const auto previous = ComboStepTint(step - 1);
        const auto tint = ComboStepTint(step);
        CHECK(tint[0] < previous[0]);
        CHECK(tint[1] < previous[1]);
        CHECK(tint[2] > previous[2]);
    }

    CHECK(ComboStepTint(9) == ComboStepTint(4));
}
