// Web build: no Crypto++ in the browser, so version 14 (encrypted) BMD models can't be read there.
// Our game data has none (only versions 10 and 12), so the loader simply never asks; if one shows up,
// it's refused like a damaged file instead of crashing.
#include "Render/Models/BmdModulusCipher.h"

namespace Render::Models::BmdModulusCipher
{
bool DecryptInPlace(std::uint8_t* /*payload*/, std::size_t /*size*/)
{
    return false;
}
} // namespace Render::Models::BmdModulusCipher
