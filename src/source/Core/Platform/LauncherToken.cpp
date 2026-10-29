#include "LauncherToken.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <array>

namespace Core::Platform
{
namespace
{
std::string Signature(std::string_view seconds, std::string_view key)
{
    const std::string message = "mu-launcher|" + std::string(seconds);
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length = 0;
    if (HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), reinterpret_cast<const unsigned char*>(message.data()), message.size(), digest.data(), &length) == nullptr
        || length < 16)
    {
        return {};
    }

    constexpr char hex[] = "0123456789abcdef";
    std::string text;
    text.reserve(32);
    for (std::size_t i = 0; i < 16; ++i)
    {
        text.push_back(hex[digest[i] >> 4]);
        text.push_back(hex[digest[i] & 0x0F]);
    }

    return text;
}
}

std::string CreateLauncherToken(std::int64_t unixSeconds, std::string_view key)
{
    const std::string seconds = std::to_string(unixSeconds);
    return seconds + "-" + Signature(seconds, key);
}

bool IsLauncherTokenValid(std::wstring_view token, std::int64_t nowUnixSeconds, std::string_view key)
{
    const auto dash = token.find(L'-');
    if (dash == std::wstring_view::npos || dash == 0 || dash > 19)
    {
        return false;
    }

    std::string seconds;
    std::int64_t value = 0;
    for (const wchar_t c : token.substr(0, dash))
    {
        if (c < L'0' || c > L'9')
        {
            return false;
        }

        seconds.push_back(static_cast<char>(c));
        value = value * 10 + (c - L'0');
    }

    std::string signature;
    for (const wchar_t c : token.substr(dash + 1))
    {
        if (c > 0x7F)
        {
            return false;
        }

        signature.push_back(static_cast<char>(c));
    }

    const std::int64_t age = nowUnixSeconds > value ? nowUnixSeconds - value : value - nowUnixSeconds;
    const std::string expected = Signature(seconds, key);
    return age <= kLauncherTokenValidSeconds && !expected.empty() && signature.size() == expected.size()
        && CRYPTO_memcmp(signature.data(), expected.data(), expected.size()) == 0;
}
}
