// Tests of the point mode Castle Siege scoreboard packet (C2 FB 01), which OpenMU's
// ShowCastleSiegeScoreboardPlugIn sends: the guilds by points, the castle holder, the hero's guild,
// and the timers the client counts down itself between packets.

#include "doctest.h"

#include "GameLogic/Events/CastleSiegeScoreboard.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace GameLogic::CastleSiege;

namespace
{
struct Guild
{
    std::string name;
    std::int32_t points;
    std::uint8_t seals;
    std::uint8_t flags;
};

std::vector<std::uint8_t> MakePacket(std::uint8_t state, std::uint16_t remaining, std::uint16_t nextTick, const std::vector<Guild>& guilds)
{
    std::vector<std::uint8_t> packet{0xC2, 0, 0, kScoreboardPacketCode, kScoreboardPacketSubCode, state,
                                     static_cast<std::uint8_t>(remaining), static_cast<std::uint8_t>(remaining >> 8),
                                     static_cast<std::uint8_t>(nextTick), static_cast<std::uint8_t>(nextTick >> 8),
                                     static_cast<std::uint8_t>(guilds.size())};
    for (const auto& guild : guilds)
    {
        for (std::size_t i = 0; i < 8; ++i)
            packet.push_back(i < guild.name.size() ? static_cast<std::uint8_t>(guild.name[i]) : 0);
        for (int shift = 0; shift < 32; shift += 8)
            packet.push_back(static_cast<std::uint8_t>(static_cast<std::uint32_t>(guild.points) >> shift));
        packet.push_back(guild.seals);
        packet.push_back(guild.flags);
    }

    packet[1] = static_cast<std::uint8_t>(packet.size() >> 8);
    packet[2] = static_cast<std::uint8_t>(packet.size());
    return packet;
}
}

TEST_CASE("A scoreboard packet lists the guilds with their points and marks")
{
    Scoreboard scoreboard;
    const auto packet = MakePacket(1, 2472, 221, {{"Phoenix", 26, 2, 0x01}, {"Valhalla", 23, 1, 0x02}});

    REQUIRE(scoreboard.Apply(packet, 1000));

    CHECK(scoreboard.State() == ScoreboardState::Running);
    REQUIRE(scoreboard.Guilds().size() == 2);
    CHECK(scoreboard.Guilds()[0].name == "Phoenix");
    CHECK(scoreboard.Guilds()[0].points == 26);
    CHECK(scoreboard.Guilds()[0].holdsCastle);
    CHECK_FALSE(scoreboard.Guilds()[0].isOwnGuild);
    CHECK(scoreboard.Guilds()[1].name == "Valhalla");
    CHECK(scoreboard.Guilds()[1].seals == 1);
    CHECK(scoreboard.Guilds()[1].isOwnGuild);
}

TEST_CASE("The timers count down from the last packet")
{
    Scoreboard scoreboard;
    REQUIRE(scoreboard.Apply(MakePacket(1, 600, 120, {}), 5000));

    CHECK(scoreboard.RemainingSeconds(5000) == 600);
    CHECK(scoreboard.RemainingSeconds(65'000) == 540);
    CHECK(scoreboard.NextDefenseTickSeconds(65'000) == 60);
    CHECK(scoreboard.NextDefenseTickSeconds(500'000) == 0);
    CHECK(scoreboard.RemainingSeconds(10'000'000) == 0);
}

TEST_CASE("No defense tick is coming when the server says so")
{
    Scoreboard scoreboard;
    REQUIRE(scoreboard.Apply(MakePacket(1, 600, 0xFFFF, {}), 0));

    CHECK(scoreboard.NextDefenseTickSeconds(0) == -1);
}

TEST_CASE("The final scores show for a while, a forgotten running board goes stale")
{
    Scoreboard scoreboard;
    REQUIRE(scoreboard.Apply(MakePacket(2, 0, 0xFFFF, {{"Phoenix", 30, 3, 0}}), 0));
    CHECK_FALSE(scoreboard.IsExpired(kFinalScoresDisplayMs - 1));
    CHECK(scoreboard.IsExpired(kFinalScoresDisplayMs));

    REQUIRE(scoreboard.Apply(MakePacket(1, 10, 0xFFFF, {}), 0));
    CHECK_FALSE(scoreboard.IsExpired(10'000 + kStaleAfterBattleMs - 1));
    CHECK(scoreboard.IsExpired(10'000 + kStaleAfterBattleMs));

    REQUIRE(scoreboard.Apply(MakePacket(0, 0, 0xFFFF, {}), 0));
    CHECK(scoreboard.IsExpired(0));
}

TEST_CASE("A malformed packet changes nothing")
{
    Scoreboard scoreboard;
    REQUIRE(scoreboard.Apply(MakePacket(1, 600, 120, {{"Phoenix", 26, 2, 0x01}}), 0));
    const auto version = scoreboard.Version();

    auto truncated = MakePacket(1, 600, 120, {{"Ronin", 10, 1, 0}});
    truncated.pop_back();
    CHECK_FALSE(scoreboard.Apply(truncated, 0));

    auto unknownState = MakePacket(7, 600, 120, {});
    CHECK_FALSE(scoreboard.Apply(unknownState, 0));

    auto otherSubCode = MakePacket(1, 600, 120, {});
    otherSubCode[4] = 0x02;
    CHECK_FALSE(scoreboard.Apply(otherSubCode, 0));

    CHECK(scoreboard.Version() == version);
    CHECK(scoreboard.Guilds()[0].name == "Phoenix");
}
