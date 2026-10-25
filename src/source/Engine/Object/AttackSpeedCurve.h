#pragma once

namespace Engine::Object
{
    // Attack/magic speed bonus to an attack animation's PlaySpeed (keyframes per animation tick):
    // the original linear `speed * slope`, without a cap. At the attack speeds of reset servers
    // (2,000+) a swing then plays within one or two ticks and the character keeps the hit pose
    // while the casts follow each other, like the clients those servers are played with. The
    // cast rate itself stays limited elsewhere: SendRequestMagic sends a targeted skill at most
    // every 300 ms. This replaces hand-tuned bands which made 509..750 play slower than 508 and
    // then jumped.
    constexpr float kAttackSpeedSlope = 0.004f;
    constexpr float kCastSpeedSlope = 0.002f;

    inline float AttackSpeedAnimationBonus(int speed, float slope)
    {
        return static_cast<float>(speed > 0 ? speed : 0) * slope;
    }
}
