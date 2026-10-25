// Tests of the swing classification of the player actions.
//
// A position packet from the server (the basic weapon skills move the caster onto the target on every
// cast) must not cut a genuine swing short, but must still stop a character standing or walking in one
// of the stance animations that share the attack enum range.

#include "doctest.h"

#include "Engine/Object/PlayerActionState.h"

using Engine::Object::IsSwingAction;

TEST_CASE("Weapon skill swings count as swings")
{
    CHECK(IsSwingAction(PLAYER_ATTACK_FIST));
    CHECK(IsSwingAction(PLAYER_ATTACK_SKILL_SWORD1));
    CHECK(IsSwingAction(PLAYER_ATTACK_SKILL_WHEEL));
    CHECK(IsSwingAction(PLAYER_ATTACK_SKILL_SPEAR));
    CHECK(IsSwingAction(PLAYER_FENRIR_ATTACK));
    CHECK(IsSwingAction(PLAYER_RIDE_SKILL));
}

TEST_CASE("Stance animations inside the attack range are no swings")
{
    CHECK_FALSE(IsSwingAction(PLAYER_STOP_TWO_HAND_SWORD_TWO));
    CHECK_FALSE(IsSwingAction(PLAYER_DARKLORD_STAND));
    CHECK_FALSE(IsSwingAction(PLAYER_FENRIR_RUN));
    CHECK_FALSE(IsSwingAction(PLAYER_RAGE_FENRIR_WALK));
}

TEST_CASE("Actions outside the attack range are no swings")
{
    CHECK_FALSE(IsSwingAction(PLAYER_STOP_MALE));
    CHECK_FALSE(IsSwingAction(PLAYER_SKILL_SLEEP));
}
