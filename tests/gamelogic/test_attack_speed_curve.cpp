// Tests of the attack speed animation bonus.
//
// It is the original linear formula at every speed: the replaced bands made 509..750 slower than
// 508 and then jumped, and reset servers expect swings to keep getting faster with attack speed.

#include "doctest.h"

#include "Engine/Object/AttackSpeedCurve.h"

#include <initializer_list>

using Engine::Object::AttackSpeedAnimationBonus;
using Engine::Object::kAttackSpeedSlope;
using Engine::Object::kCastSpeedSlope;

TEST_CASE("The bonus is the original linear formula")
{
    CHECK(AttackSpeedAnimationBonus(0, kAttackSpeedSlope) == doctest::Approx(0.0f));
    CHECK(AttackSpeedAnimationBonus(200, kAttackSpeedSlope) == doctest::Approx(0.8f));
    CHECK(AttackSpeedAnimationBonus(2240, kAttackSpeedSlope) == doctest::Approx(8.96f));
    CHECK(AttackSpeedAnimationBonus(600, kCastSpeedSlope) == doctest::Approx(1.2f));
}

TEST_CASE("Negative speeds give no bonus")
{
    CHECK(AttackSpeedAnimationBonus(-50, kAttackSpeedSlope) == doctest::Approx(0.0f));
}

TEST_CASE("The bonus never drops with more speed")
{
    for (const float slope : std::initializer_list<float>{kAttackSpeedSlope, kCastSpeedSlope})
    {
        float previous = AttackSpeedAnimationBonus(0, slope);
        for (int speed = 1; speed <= 65535; ++speed)
        {
            const float bonus = AttackSpeedAnimationBonus(speed, slope);
            REQUIRE(bonus > previous);
            previous = bonus;
        }
    }
}
