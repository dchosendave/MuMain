// Web build: no OpenSSL. Saved credentials ("RememberMe"/"SavePassword") would
// end up in browser storage, which this blob encryption cannot protect anyway,
// so encryption fails and the client simply does not store them.
#include "stdafx.h"

#include "Core/Platform/PlatformCrypto.h"

bool mu_encrypt_blob(const void* /*input*/, std::size_t /*inputSize*/, std::vector<std::uint8_t>& output)
{
    output.clear();
    return false;
}

bool mu_decrypt_blob(const void* /*input*/, std::size_t /*inputSize*/, std::vector<std::uint8_t>& output)
{
    output.clear();
    return false;
}
