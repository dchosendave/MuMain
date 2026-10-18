// Web build: ConnectionManager_* exports of MUnique.Client.Library over browser WebSockets.
//
// Each connection is a WebSocket to MuWeb.Gateway at <gateway>/ws/<port>. The gateway knows
// which OpenMU host serves that port and applies the MU encryption, so the host argument and
// the isEncrypted flag of ConnectionManager_Connect are not needed here.
//
// Like the native library, Connect returns a handle at once. Sends made before the socket is
// open are queued; if it never opens, the disconnect callback fires.
#include "stdafx.h"

#include "Dotnet/web/WebPacketBuffer.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <emscripten.h>
#include <emscripten/websocket.h>

#include "Core/Utilities/Log/MuLogger.h"

namespace mu::web
{
namespace
{
using PacketReceivedCallback = void (*)(int32_t, int32_t, BYTE*);
using DisconnectedCallback = void (*)(int32_t);
using LogCallback = void (*)(BYTE, const char*);

struct WebConnection
{
    EMSCRIPTEN_WEBSOCKET_T socket = 0;
    bool open = false;
    bool receiving = false;
    bool closed = false;
    PacketReceivedCallback onPacketReceived = nullptr;
    DisconnectedCallback onDisconnected = nullptr;
    std::deque<std::vector<std::uint8_t>> pendingSends;
    std::deque<std::vector<std::uint8_t>> pendingReceives;
};

std::map<int32_t, WebConnection> g_connections;
int32_t g_maxHandle = 0;

// Gateway base URL: Module.muGatewayUrl if the page sets it, otherwise the page's own origin.
// (No regex literals here: EM_JS bodies are stringified by the preprocessor.)
EM_JS(char*, mu_web_gateway_url, (), {
    var url = (typeof Module !== 'undefined' && Module['muGatewayUrl']) ? Module['muGatewayUrl']
        : ((location.protocol === 'https:' ? 'wss:' : 'ws:') + String.fromCharCode(47, 47) + location.host);
    return stringToNewUTF8(url);
});

std::string GatewayUrl()
{
    char* url = mu_web_gateway_url();
    std::string result(url);
    free(url);
    while (!result.empty() && result.back() == '/')
    {
        result.pop_back();
    }

    return result;
}

WebConnection* Find(int32_t handle)
{
    const auto it = g_connections.find(handle);
    return it == g_connections.end() ? nullptr : &it->second;
}

void SendNow(WebConnection& connection, const std::vector<std::uint8_t>& packet)
{
    emscripten_websocket_send_binary(connection.socket, const_cast<std::uint8_t*>(packet.data()),
                                     static_cast<uint32_t>(packet.size()));
}

void Deliver(int32_t handle, WebConnection& connection)
{
    while (connection.receiving && !connection.pendingReceives.empty())
    {
        auto packet = std::move(connection.pendingReceives.front());
        connection.pendingReceives.pop_front();
        connection.onPacketReceived(handle, static_cast<int32_t>(packet.size()), packet.data());
        if (Find(handle) == nullptr)
        {
            return; // the callback disconnected
        }
    }
}

void NotifyClosed(int32_t handle)
{
    WebConnection* connection = Find(handle);
    if (connection == nullptr || connection->closed)
    {
        return;
    }

    connection->closed = true;
    const auto onDisconnected = connection->onDisconnected;
    emscripten_websocket_delete(connection->socket);
    g_connections.erase(handle);
    mu::log::Get("dotnet")->info("NET: web connection handle={} closed", handle);
    if (onDisconnected != nullptr)
    {
        onDisconnected(handle);
    }
}

EM_BOOL OnOpen(int, const EmscriptenWebSocketOpenEvent*, void* userData)
{
    const auto handle = static_cast<int32_t>(reinterpret_cast<intptr_t>(userData));
    if (WebConnection* connection = Find(handle))
    {
        connection->open = true;
        for (const auto& packet : connection->pendingSends)
        {
            SendNow(*connection, packet);
        }

        connection->pendingSends.clear();
    }

    return EM_TRUE;
}

EM_BOOL OnMessage(int, const EmscriptenWebSocketMessageEvent* event, void* userData)
{
    const auto handle = static_cast<int32_t>(reinterpret_cast<intptr_t>(userData));
    WebConnection* connection = Find(handle);
    if (connection == nullptr || event->isText)
    {
        return EM_TRUE;
    }

    // The gateway sends exactly one packet per message.
    connection->pendingReceives.emplace_back(event->data, event->data + event->numBytes);
    Deliver(handle, *connection);
    return EM_TRUE;
}

EM_BOOL OnClose(int, const EmscriptenWebSocketCloseEvent* event, void* userData)
{
    const auto handle = static_cast<int32_t>(reinterpret_cast<intptr_t>(userData));
    mu::log::Get("dotnet")->info("NET: web socket handle={} closed by peer (code {})", handle, event->code);
    NotifyClosed(handle);
    return EM_TRUE;
}

EM_BOOL OnError(int, const EmscriptenWebSocketErrorEvent*, void* userData)
{
    const auto handle = static_cast<int32_t>(reinterpret_cast<intptr_t>(userData));
    mu::log::Get("dotnet")->error("NET: web socket handle={} failed", handle);
    NotifyClosed(handle);
    return EM_TRUE;
}

// --- MUnique.Client.Library exports -------------------------------------------------------

void SetLogCallback(LogCallback)
{
    // Managed logging only; the web network layer logs through mu::log directly.
}

int32_t Connect(const char16_t* host, int32_t port, BYTE /*isEncrypted*/, PacketReceivedCallback onPacketReceived,
                DisconnectedCallback onDisconnected)
{
    if (!emscripten_websocket_is_supported())
    {
        mu::log::Get("dotnet")->error("NET: WebSockets are not supported by this browser");
        return -1;
    }

    const std::string url = GatewayUrl() + "/ws/" + std::to_string(port);
    EmscriptenWebSocketCreateAttributes attributes;
    emscripten_websocket_init_create_attributes(&attributes);
    attributes.url = url.c_str();
    attributes.protocols = nullptr;
    attributes.createOnMainThread = EM_TRUE;
    const EMSCRIPTEN_WEBSOCKET_T socket = emscripten_websocket_new(&attributes);
    if (socket <= 0)
    {
        mu::log::Get("dotnet")->error("NET: could not create a web socket to {} ({})", url, socket);
        return -1;
    }

    const int32_t handle = ++g_maxHandle;
    WebConnection& connection = g_connections[handle];
    connection.socket = socket;
    connection.onPacketReceived = onPacketReceived;
    connection.onDisconnected = onDisconnected;

    void* userData = reinterpret_cast<void*>(static_cast<intptr_t>(handle));
    emscripten_websocket_set_onopen_callback(socket, userData, OnOpen);
    emscripten_websocket_set_onmessage_callback(socket, userData, OnMessage);
    emscripten_websocket_set_onclose_callback(socket, userData, OnClose);
    emscripten_websocket_set_onerror_callback(socket, userData, OnError);

    mu::log::Get("dotnet")->info("NET: web connection handle={} -> {} (requested host {})", handle, url,
                                Utf16ToUtf8(host));
    return handle;
}

void Send(int32_t handle, const BYTE* data, int32_t count)
{
    WebConnection* connection = Find(handle);
    if (connection == nullptr || data == nullptr || count < 3)
    {
        mu::log::Get("dotnet")->error("NET: Send skipped; connection handle={} not found", handle);
        return;
    }

    std::vector<std::uint8_t> packet(data, data + count);
    SetPacketSize(packet.data(), packet.size());
    if (connection->open)
    {
        SendNow(*connection, packet);
    }
    else
    {
        connection->pendingSends.push_back(std::move(packet));
    }
}

void BeginReceive(int32_t handle)
{
    if (WebConnection* connection = Find(handle))
    {
        connection->receiving = true;
        Deliver(handle, *connection);
    }
}

void Disconnect(int32_t handle)
{
    if (WebConnection* connection = Find(handle))
    {
        emscripten_websocket_close(connection->socket, 1000, "client disconnect");
        // Like the native library, the disconnect callback follows asynchronously (OnClose).
    }
}

// Hand-written in MUnique.Client.Library (ConnectionManager.*.Custom.cs).
void SendLogin(int32_t handle, const char16_t* username, const char16_t* password, uint32_t tickCount,
               const BYTE* clientVersion, const BYTE* clientSerial)
{
    const std::string user = Utf16ToUtf8(username);
    const std::string pass = Utf16ToUtf8(password);
    if (user.size() > 10 || pass.size() > 20 || clientVersion == nullptr || clientSerial == nullptr)
    {
        mu::log::Get("dotnet")->error("NET: Login packet staging failed, handle={}: invalid credentials", handle);
        return;
    }

    // LoginLongPassword; user name and password are Xor3 "encrypted" over their full field length.
    PacketBuffer packet(0xC3, 0xF1, 0x01, 60);
    packet.WriteBytes(4, user.data(), user.size());
    packet.WriteBytes(14, pass.data(), pass.size());
    packet.Xor3(4, 10);
    packet.Xor3(14, 20);
    packet.WriteU32Be(34, tickCount);
    packet.WriteBytes(38, clientVersion, 5);
    packet.WriteBytes(43, clientSerial, 16);
    SubmitPacket(handle, packet);
}

void SendAuthenticateExt(int32_t handle, uint16_t roomId, uint32_t token)
{
    // ChatServer Authenticate: C1 00, room id (LE) at 4, token as decimal text at 6..15, Xor3 from 6.
    PacketBuffer packet(0xC1, 0x00, -1, 16);
    packet.WriteU16Le(4, roomId);
    const std::string tokenText = std::to_string(token);
    packet.WriteBytes(6, tokenText.data(), std::min<std::size_t>(tokenText.size(), 10));
    packet.Xor3(6);
    SubmitPacket(handle, packet);
}

void SendChatMessageExt(int32_t handle, BYTE senderIndex, const char16_t* message)
{
    // ChatServer ChatMessage: C1 04, sender index at 3, message length at 4, message at 5, Xor3 from 5.
    const std::string text = Utf16ToUtf8(message);
    PacketBuffer packet(0xC1, 0x04, -1, 5 + text.size());
    packet.WriteByte(3, senderIndex);
    packet.WriteByte(4, static_cast<std::uint8_t>(text.size()));
    packet.WriteBytes(5, text.data(), text.size());
    packet.Xor3(5);
    SubmitPacket(handle, packet);
}

const NetworkSymbol kHandWrittenSymbols[] = {
    {"ConnectionManager_SetLogCallback", reinterpret_cast<void*>(&SetLogCallback)},
    {"ConnectionManager_Connect", reinterpret_cast<void*>(&Connect)},
    {"ConnectionManager_Send", reinterpret_cast<void*>(&Send)},
    {"ConnectionManager_BeginReceive", reinterpret_cast<void*>(&BeginReceive)},
    {"ConnectionManager_Disconnect", reinterpret_cast<void*>(&Disconnect)},
    {"ConnectionManager_SendLogin", reinterpret_cast<void*>(&SendLogin)},
    {"ConnectionManager_SendAuthenticateExt", reinterpret_cast<void*>(&SendAuthenticateExt)},
    {"ConnectionManager_SendChatMessageExt", reinterpret_cast<void*>(&SendChatMessageExt)},
};
} // namespace

void SubmitPacket(int32_t handle, const PacketBuffer& packet)
{
    const auto& bytes = packet.Bytes();
    Send(handle, bytes.data(), static_cast<int32_t>(bytes.size()));
}

void* FindNetworkSymbol(const char* name)
{
    for (const auto& symbol : kHandWrittenSymbols)
    {
        if (std::strcmp(symbol.name, name) == 0)
        {
            return symbol.address;
        }
    }

    for (std::size_t i = 0; i < kGeneratedPacketSymbolCount; ++i)
    {
        if (std::strcmp(kGeneratedPacketSymbols[i].name, name) == 0)
        {
            return kGeneratedPacketSymbols[i].address;
        }
    }

    return nullptr;
}

void* NetworkLibraryHandle()
{
    static int marker = 0;
    return &marker;
}
} // namespace mu::web
