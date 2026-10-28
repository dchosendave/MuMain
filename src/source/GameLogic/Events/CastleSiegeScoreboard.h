#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace GameLogic::CastleSiege
{
    // The scoreboard of a point mode Castle Siege battle, as the server last sent it with the custom
    // packet C2 FB 01 (OpenMU's ShowCastleSiegeScoreboardPlugIn). Guilds score with seals, switch
    // holder kills and time holding the castle; the one with the most points takes the castle.
    constexpr std::uint8_t kScoreboardPacketCode = 0xFB;
    constexpr std::uint8_t kScoreboardPacketSubCode = 0x01;

    // How long the final scores stay on screen after the battle.
    constexpr std::uint32_t kFinalScoresDisplayMs = 30'000;

    // A running board which outlived its battle by this long is stale, e.g. the hero left the siege
    // map and missed the final scores.
    constexpr std::uint32_t kStaleAfterBattleMs = 60'000;

    enum class ScoreboardState : std::uint8_t
    {
        Hidden = 0,
        Running = 1,
        Final = 2,
    };

    struct ScoreboardGuild
    {
        std::string name;
        int points = 0;
        int seals = 0;
        bool holdsCastle = false;
        bool isOwnGuild = false;
    };

    class Scoreboard
    {
    public:
        // Reads a scoreboard packet (the whole packet, header included) received at `nowMs`.
        // A malformed packet changes nothing and returns false.
        bool Apply(std::span<const std::uint8_t> packet, std::uint32_t nowMs);

        void Clear();

        ScoreboardState State() const { return m_state; }

        const std::vector<ScoreboardGuild>& Guilds() const { return m_guilds; }

        // Counts up with every applied packet, so a view can tell when to refresh.
        std::uint32_t Version() const { return m_version; }

        // The seconds left of the battle at `nowMs`, counted down from the last packet.
        int RemainingSeconds(std::uint32_t nowMs) const;

        // The seconds until the castle holder's next defense tick; -1 if none is coming.
        int NextDefenseTickSeconds(std::uint32_t nowMs) const;

        // Nothing to show any more: the final scores were shown long enough, or the battle is over
        // without the final scores having arrived.
        bool IsExpired(std::uint32_t nowMs) const;

    private:
        int CountDown(int seconds, std::uint32_t nowMs) const;

        ScoreboardState m_state = ScoreboardState::Hidden;
        std::vector<ScoreboardGuild> m_guilds;
        int m_remainingSeconds = 0;
        int m_nextDefenseTickSeconds = -1;
        std::uint32_t m_receivedMs = 0;
        std::uint32_t m_version = 0;
    };

    extern Scoreboard g_Scoreboard;
}
