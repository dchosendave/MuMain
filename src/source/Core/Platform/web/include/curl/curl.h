// Web build only: declarations the in-game shop headers need from libcurl.
// There is no libcurl in the browser; CurlFileDownloaderWeb.cpp reports every
// download as failed, so nothing here is ever called.
#pragma once

#include <cstdint>

typedef void CURL;
typedef std::int64_t curl_off_t;
