#include "stdafx.h"

#include "Dotnet/web/WebPacketBuffer.h"

#include <cstring>

namespace mu::web
{
namespace
{
constexpr std::uint8_t Xor3Keys[] = {0xFC, 0xCF, 0xAB};

void AppendUtf8(std::string& out, char32_t codePoint)
{
    if (codePoint < 0x80)
    {
        out.push_back(static_cast<char>(codePoint));
    }
    else if (codePoint < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else if (codePoint < 0x10000)
    {
        out.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
}
} // namespace

std::string Utf16ToUtf8(const char16_t* text)
{
    std::string out;
    if (text == nullptr)
    {
        return out;
    }

    for (const char16_t* p = text; *p != 0; ++p)
    {
        const char16_t unit = *p;
        if (unit >= 0xD800 && unit <= 0xDBFF && p[1] >= 0xDC00 && p[1] <= 0xDFFF)
        {
            AppendUtf8(out, 0x10000 + ((static_cast<char32_t>(unit) - 0xD800) << 10) + (p[1] - 0xDC00));
            ++p;
        }
        else if (unit >= 0xD800 && unit <= 0xDFFF)
        {
            AppendUtf8(out, 0xFFFD);
        }
        else
        {
            AppendUtf8(out, unit);
        }
    }

    return out;
}

PacketBuffer::PacketBuffer(std::uint8_t headerType, std::uint8_t code, int subCode, std::size_t size)
    : _data(size, 0)
{
    const bool longHeader = headerType == 0xC2 || headerType == 0xC4;
    const std::size_t codeIndex = longHeader ? 3 : 2;
    if (_data.size() <= codeIndex)
    {
        _data.resize(codeIndex + 1);
    }

    _data[0] = headerType;
    _data[codeIndex] = code;
    if (subCode >= 0 && _data.size() > codeIndex + 1)
    {
        _data[codeIndex + 1] = static_cast<std::uint8_t>(subCode);
    }

    SetPacketSize(_data.data(), _data.size());
}

void PacketBuffer::WriteByte(std::size_t index, std::uint8_t value)
{
    _data[index] = value;
}

void PacketBuffer::WriteBits(std::size_t index, std::uint8_t value, int bits, int leftShifted)
{
    const auto bitMask = static_cast<std::uint8_t>(((1u << bits) - 1u) << leftShifted);
    _data[index] = static_cast<std::uint8_t>((_data[index] & ~bitMask) | ((value << leftShifted) & bitMask));
}

void PacketBuffer::WriteBoolean(std::size_t index, bool value, int leftShifted)
{
    WriteBits(index, value ? 1 : 0, 1, leftShifted);
}

void PacketBuffer::WriteU16Le(std::size_t index, std::uint16_t value)
{
    _data[index] = static_cast<std::uint8_t>(value);
    _data[index + 1] = static_cast<std::uint8_t>(value >> 8);
}

void PacketBuffer::WriteU16Be(std::size_t index, std::uint16_t value)
{
    _data[index] = static_cast<std::uint8_t>(value >> 8);
    _data[index + 1] = static_cast<std::uint8_t>(value);
}

void PacketBuffer::WriteU32Le(std::size_t index, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        _data[index + i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

void PacketBuffer::WriteU32Be(std::size_t index, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        _data[index + i] = static_cast<std::uint8_t>(value >> (8 * (3 - i)));
    }
}

void PacketBuffer::WriteU64Le(std::size_t index, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        _data[index + i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

void PacketBuffer::WriteU64Be(std::size_t index, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        _data[index + i] = static_cast<std::uint8_t>(value >> (8 * (7 - i)));
    }
}

void PacketBuffer::WriteBytes(std::size_t index, const void* data, std::size_t length)
{
    if (length > 0 && data != nullptr)
    {
        std::memcpy(_data.data() + index, data, length);
    }
}

bool PacketBuffer::WriteFixedString(std::size_t index, std::size_t length, const char16_t* text)
{
    const std::string utf8 = Utf16ToUtf8(text);
    if (utf8.size() > length)
    {
        return false;
    }

    std::memset(_data.data() + index, 0, length);
    WriteBytes(index, utf8.data(), utf8.size());
    return true;
}

bool PacketBuffer::WriteFixedBinary(std::size_t index, std::size_t length, const std::uint8_t* data, std::uint32_t dataLength)
{
    if (dataLength > length || (data == nullptr && dataLength > 0))
    {
        return false;
    }

    WriteBytes(index, data, dataLength);
    return true;
}

void PacketBuffer::Xor3(std::size_t startOffset, std::size_t length)
{
    const std::size_t end = length > _data.size() - startOffset ? _data.size() : startOffset + length;
    for (std::size_t i = startOffset; i < end; ++i)
    {
        _data[i] ^= Xor3Keys[(i - startOffset) % 3];
    }
}

void SetPacketSize(std::uint8_t* packet, std::size_t length)
{
    if (packet[0] == 0xC1 || packet[0] == 0xC3)
    {
        packet[1] = static_cast<std::uint8_t>(length);
    }
    else
    {
        packet[1] = static_cast<std::uint8_t>(length >> 8);
        packet[2] = static_cast<std::uint8_t>(length);
    }
}
} // namespace mu::web
