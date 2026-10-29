#include <doctest.h>

#include "Core/Platform/LauncherToken.h"

#include <string>

namespace
{
constexpr const char* kKey = "mu-online-launcher-v1";
constexpr std::int64_t kNow = 1800000000;

std::wstring Wide(const std::string& text)
{
    return std::wstring(text.begin(), text.end());
}
}

TEST_CASE("A launcher token is accepted for two minutes either way")
{
    const auto token = Wide(Core::Platform::CreateLauncherToken(kNow, kKey));

    CHECK(Core::Platform::IsLauncherTokenValid(token, kNow, kKey));
    CHECK(Core::Platform::IsLauncherTokenValid(token, kNow + 120, kKey));
    CHECK(Core::Platform::IsLauncherTokenValid(token, kNow - 120, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(token, kNow + 121, kKey));
}

TEST_CASE("A changed, malformed or foreign launcher token is refused")
{
    auto token = Wide(Core::Platform::CreateLauncherToken(kNow, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(token, kNow, "another key"));

    token.back() = token.back() == L'a' ? L'b' : L'a';
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(token, kNow, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(L"", kNow, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(L"garbage", kNow, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(L"-abc", kNow, kKey));
    CHECK_FALSE(Core::Platform::IsLauncherTokenValid(L"12x4-abc", kNow, kKey));
}

TEST_CASE("The launcher token is the one MuLauncher makes")
{
    // The same vector is in MuLauncher's LauncherTests.LaunchToken_MatchesTheValueMuMainChecks.
    CHECK(Core::Platform::CreateLauncherToken(kNow, kKey) == "1800000000-368e86d91336c5756d29cfed2cda60b4");
}
