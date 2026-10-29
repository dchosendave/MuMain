#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Core::Platform
{
// The token MuLauncher passes as "/l<token>", so a player build (MU_REQUIRE_LAUNCHER) only starts from
// the launcher, which keeps the game files up to date. Format: "<unix seconds>-<hex of the first 16
// bytes of HMAC-SHA256(key, "mu-launcher|" + seconds)>", accepted for kLauncherTokenValidSeconds either
// way. The key is in both programs, so this guards against outdated files, not against cheaters.
// MuLauncher.Core/LaunchToken.cs makes the same token.
inline constexpr std::int64_t kLauncherTokenValidSeconds = 120;

[[nodiscard]] std::string CreateLauncherToken(std::int64_t unixSeconds, std::string_view key);

[[nodiscard]] bool IsLauncherTokenValid(std::wstring_view token, std::int64_t nowUnixSeconds, std::string_view key);
}
