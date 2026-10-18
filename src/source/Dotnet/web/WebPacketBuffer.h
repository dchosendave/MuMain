// Web build: the network layer that replaces MUnique.Client.Library.
//
// The native client loads ConnectionManager_* and Send* from the .NET Native AOT
// library. In the browser those symbols are implemented here in C++ and talk to
// MuWeb.Gateway over WebSockets; the gateway applies the MU encryption, so packets
// leave the browser as plain packets. See tools/gen_web_packets.py for Send*.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mu::web
{
struct NetworkSymbol
{
    const char* name;
    void* address;
};

// Resolves an export of MUnique.Client.Library by name; nullptr when unknown.
void* FindNetworkSymbol(const char* name);

// A non-null handle standing in for the loaded library, so the native
// "is the library available" checks pass.
void* NetworkLibraryHandle();

// UTF-16 -> UTF-8 like .NET's Encoding.UTF8 (unpaired surrogates become U+FFFD).
std::string Utf16ToUtf8(const char16_t* text);

// A packet being built, zero initialized, with its header (type, length, code, sub code) set.
// Mirrors the generated ref structs of MUnique.OpenMU.Network.Packets.
class PacketBuffer
{
public:
    PacketBuffer(std::uint8_t headerType, std::uint8_t code, int subCode, std::size_t size);

    void WriteByte(std::size_t index, std::uint8_t value);
    void WriteBits(std::size_t index, std::uint8_t value, int bits, int leftShifted);
    void WriteBoolean(std::size_t index, bool value, int leftShifted);
    void WriteU16Le(std::size_t index, std::uint16_t value);
    void WriteU16Be(std::size_t index, std::uint16_t value);
    void WriteU32Le(std::size_t index, std::uint32_t value);
    void WriteU32Be(std::size_t index, std::uint32_t value);
    void WriteU64Le(std::size_t index, std::uint64_t value);
    void WriteU64Be(std::size_t index, std::uint64_t value);
    void WriteBytes(std::size_t index, const void* data, std::size_t length);

    // Like WriteString on a fixed slice: fails (the packet is not sent) when the UTF-8 text does not fit.
    bool WriteFixedString(std::size_t index, std::size_t length, const char16_t* text);

    // Like Span.CopyTo into a fixed slice: fails when the source is longer than the field.
    bool WriteFixedBinary(std::size_t index, std::size_t length, const std::uint8_t* data, std::uint32_t dataLength);

    // XORs `length` bytes from startOffset (default: to the end) with the Xor3 keys (Xor3Encryptor).
    void Xor3(std::size_t startOffset, std::size_t length = static_cast<std::size_t>(-1));

    const std::vector<std::uint8_t>& Bytes() const { return _data; }
    std::vector<std::uint8_t>& Bytes() { return _data; }

private:
    std::vector<std::uint8_t> _data;
};

// Sends a finished packet over the connection with this handle.
void SubmitPacket(std::int32_t handle, const PacketBuffer& packet);

// Sets the size field of a raw packet from its length (ArrayExtensions.SetPacketSize).
void SetPacketSize(std::uint8_t* packet, std::size_t length);

extern const NetworkSymbol kGeneratedPacketSymbols[];
extern const std::size_t kGeneratedPacketSymbolCount;
} // namespace mu::web
