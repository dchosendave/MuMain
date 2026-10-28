#include "stdafx.h"
#include "GameLogic/Events/CastleSiegeScoreboard.h"

#include <algorithm>

namespace GameLogic::CastleSiege
{
    Scoreboard g_Scoreboard;

    namespace
    {
        // C2, size (2 bytes, big endian), code, sub code, state, remaining seconds (2), next defense
        // tick seconds (2), guild count.
        constexpr std::size_t kHeaderSize = 11;
        constexpr std::size_t kNameSize = 8;
        // Name, points (4), seals (1), flags (1).
        constexpr std::size_t kGuildSize = kNameSize + 4 + 1 + 1;
        constexpr std::size_t kMaximumGuilds = 8;
        constexpr std::uint16_t kNoDefenseTick = 0xFFFF;
        constexpr std::uint8_t kHoldsCastleFlag = 0x01;
        constexpr std::uint8_t kOwnGuildFlag = 0x02;
        constexpr std::uint32_t kMillisecondsPerSecond = 1000;

        std::uint16_t ReadUInt16LittleEndian(std::span<const std::uint8_t> bytes)
        {
            return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
        }

        std::int32_t ReadInt32LittleEndian(std::span<const std::uint8_t> bytes)
        {
            const std::uint32_t value = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
            return static_cast<std::int32_t>(value);
        }

        std::string ReadName(std::span<const std::uint8_t> bytes)
        {
            const auto end = std::find(bytes.begin(), bytes.end(), std::uint8_t{0});
            return std::string(bytes.begin(), end);
        }

        ScoreboardGuild ReadGuild(std::span<const std::uint8_t> entry)
        {
            const std::uint8_t flags = entry[kNameSize + 5];
            return {
                ReadName(entry.first(kNameSize)),
                ReadInt32LittleEndian(entry.subspan(kNameSize, 4)),
                entry[kNameSize + 4],
                (flags & kHoldsCastleFlag) != 0,
                (flags & kOwnGuildFlag) != 0,
            };
        }
    }

    bool Scoreboard::Apply(std::span<const std::uint8_t> packet, std::uint32_t nowMs)
    {
        if (packet.size() < kHeaderSize || packet[0] != 0xC2
            || packet[3] != kScoreboardPacketCode || packet[4] != kScoreboardPacketSubCode)
        {
            return false;
        }

        const std::size_t declaredSize = (static_cast<std::size_t>(packet[1]) << 8) | packet[2];
        const std::size_t guildCount = packet[10];
        const auto state = static_cast<ScoreboardState>(packet[5]);
        if (declaredSize > packet.size() || guildCount > kMaximumGuilds
            || declaredSize < kHeaderSize + guildCount * kGuildSize || state > ScoreboardState::Final)
        {
            return false;
        }

        std::vector<ScoreboardGuild> guilds;
        guilds.reserve(guildCount);
        for (std::size_t i = 0; i < guildCount; ++i)
        {
            guilds.push_back(ReadGuild(packet.subspan(kHeaderSize + i * kGuildSize, kGuildSize)));
        }

        const std::uint16_t nextTick = ReadUInt16LittleEndian(packet.subspan(8, 2));
        m_state = state;
        m_guilds = std::move(guilds);
        m_remainingSeconds = ReadUInt16LittleEndian(packet.subspan(6, 2));
        m_nextDefenseTickSeconds = nextTick == kNoDefenseTick ? -1 : nextTick;
        m_receivedMs = nowMs;
        ++m_version;
        return true;
    }

    void Scoreboard::Clear()
    {
        m_state = ScoreboardState::Hidden;
        m_guilds.clear();
        ++m_version;
    }

    int Scoreboard::RemainingSeconds(std::uint32_t nowMs) const
    {
        return CountDown(m_remainingSeconds, nowMs);
    }

    int Scoreboard::NextDefenseTickSeconds(std::uint32_t nowMs) const
    {
        return m_nextDefenseTickSeconds < 0 ? -1 : CountDown(m_nextDefenseTickSeconds, nowMs);
    }

    bool Scoreboard::IsExpired(std::uint32_t nowMs) const
    {
        const std::uint32_t elapsedMs = nowMs - m_receivedMs;
        switch (m_state)
        {
        case ScoreboardState::Final:
            return elapsedMs >= kFinalScoresDisplayMs;
        case ScoreboardState::Running:
            return elapsedMs >= static_cast<std::uint32_t>(m_remainingSeconds) * kMillisecondsPerSecond + kStaleAfterBattleMs;
        default:
            return true;
        }
    }

    int Scoreboard::CountDown(int seconds, std::uint32_t nowMs) const
    {
        const auto elapsed = static_cast<int>((nowMs - m_receivedMs) / kMillisecondsPerSecond);
        return std::max(0, seconds - elapsed);
    }
}
