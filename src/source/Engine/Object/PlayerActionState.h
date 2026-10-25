#pragma once

#include "Core/Globals/_enum.h"

namespace Engine::Object
{
    // True while a player's action is one of the attack/skill swing animations
    // (PLAYER_ATTACK_FIST .. PLAYER_RIDE_SKILL). The swing animation's playback
    // speed scales with AttackSpeed (see SetAttackSpeed in ZzzCharacter.cpp), so
    // this predicate doubles as the natural attack-cadence gate: hold off the
    // next action until the current swing finishes, and the rate follows attack
    // speed instead of any fixed timer.
    inline bool IsAttackAction(int currentAction)
    {
        return currentAction >= PLAYER_ATTACK_FIST && currentAction <= PLAYER_RIDE_SKILL;
    }

    // True while a player plays a genuine attack/skill swing. Several non-swing
    // *stance* animations (mounted idle/walk/run, two-hand-sword stance,
    // ride-horse, rage-fenrir) share the IsAttackAction() enum range; MoveHero
    // (ZzzInterface.cpp) OR-excludes exactly these four ranges when deciding
    // whether the hero may move, and so does this. Fenrir attack/skill actions
    // sit below PLAYER_FENRIR_RUN, so they still count as swings.
    inline bool IsSwingAction(int currentAction)
    {
        if (!IsAttackAction(currentAction))
        {
            return false;
        }

        return !((currentAction >= PLAYER_STOP_TWO_HAND_SWORD_TWO && currentAction <= PLAYER_RUN_TWO_HAND_SWORD_TWO) ||
            (currentAction >= PLAYER_DARKLORD_STAND && currentAction <= PLAYER_RUN_RIDE_HORSE) ||
            (currentAction >= PLAYER_FENRIR_RUN && currentAction <= PLAYER_FENRIR_WALK_ONE_LEFT) ||
            (currentAction >= PLAYER_RAGE_FENRIR_WALK && currentAction <= PLAYER_RAGE_FENRIR_STAND_ONE_LEFT));
    }

    // True while a player is sitting or holding a pose (PLAYER_SIT1 ..
    // PLAYER_POSE_FEMALE1). Used to keep these animations from being reset to the
    // stand pose when equipment or class changes.
    inline bool IsSitOrPoseAction(int currentAction)
    {
        return currentAction >= PLAYER_SIT1 && currentAction <= PLAYER_POSE_FEMALE1;
    }
}
