// Tests of the classic swing rhythm.
//
// At high attack speed the hero's swing advances in 30 Hz steps, each as far as one frame of a
// 30 FPS client, and not at all between them, whatever the rendering frame rate.

#include "doctest.h"

#include "Engine/Object/ClassicSwing.h"

using namespace Engine::Object;

namespace
{
// Advances a swing for `durationMs` at `fps` and returns how far it got.
float AdvanceSwing(int fps, int durationMs, float playSpeed)
{
    ClassicSwingClock clock;
    const float frameFactor = 25.f / static_cast<float>(fps);
    float frame = 0.f;
    clock.Steps(0);
    for (int i = 1; i <= durationMs * fps / 1000; ++i)
    {
        const auto nowMs = static_cast<std::uint32_t>(i * 1000 / fps);
        frame += ClassicSwingPlaySpeed(playSpeed, clock.Steps(nowMs), frameFactor) * frameFactor;
    }

    return frame;
}
}

TEST_CASE("Only high attack speeds use the classic rhythm")
{
    CHECK_FALSE(IsClassicSwingActive(999));
    CHECK(IsClassicSwingActive(kClassicSwingMinimumAttackSpeed));
    CHECK(IsClassicSwingActive(2240));
}

TEST_CASE("The swing advances the same distance per second at any frame rate")
{
    const float at30 = AdvanceSwing(30, 1000, 9.2f);
    CHECK(AdvanceSwing(60, 1000, 9.2f) == doctest::Approx(at30).epsilon(0.05));
    CHECK(AdvanceSwing(180, 1000, 9.2f) == doctest::Approx(at30).epsilon(0.05));
    CHECK(at30 == doctest::Approx(9.2f * 25.f).epsilon(0.05));
}

TEST_CASE("Between two 30 Hz steps the swing holds its pose")
{
    ClassicSwingClock clock;
    clock.Steps(0);
    CHECK(clock.Steps(10) == 0);
    CHECK(clock.Steps(20) == 0);
    CHECK(clock.Steps(34) == 1);
    CHECK(clock.Steps(34) == 0);
    CHECK(ClassicSwingPlaySpeed(9.2f, 0, 25.f / 180.f) == doctest::Approx(0.f));
}

TEST_CASE("A long pause doesn't jump the swing far ahead")
{
    ClassicSwingClock clock;
    clock.Steps(0);
    CHECK(clock.Steps(5000) == kClassicSwingMaximumStepsPerFrame);
}
