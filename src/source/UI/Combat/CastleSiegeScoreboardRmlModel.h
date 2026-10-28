#pragma once

#include <RmlUi/Core/Types.h>

#include <vector>

namespace mu::ui::window
{
// One guild's row: place, name and points; the theme marks the castle holder and the hero's guild.
struct CastleSiegeScoreboardLine
{
    Rml::String rank;
    Rml::String name;
    Rml::String points;
    bool holdsCastle = false;
    bool ownGuild = false;
};

struct CastleSiegeScoreboardRmlModel
{
    float boldTextPx = 0.f; // native bold text size in physical px
    float textPx = 0.f;     // native body text size in physical px

    bool final = false;          // the battle ended; these are the final scores
    Rml::String timeLeft;        // "41:12" while it runs
    Rml::String ownGuild;        // "Valhalla #2"; empty when the hero's guild isn't on the board
    Rml::String nextDefenseTick; // "03:41"; empty when no tick is coming
    std::vector<CastleSiegeScoreboardLine> lines;
};
} // namespace mu::ui::window
