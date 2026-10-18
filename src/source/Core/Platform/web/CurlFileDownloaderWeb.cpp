// Web build: the in-game shop downloader has no libcurl in the browser.
// Shop catalogs and banners are not downloaded; callers get the same failure
// code a failed curl session reports and fall back to "no shop data".
#include "stdafx.h"

#include "GameShop/FileDownloader/CurlFileDownloader.h"

WZResult CurlFileDownloader::DownloadFile(const std::wstring& url,
    const std::wstring& /*localPath*/,
    const std::wstring& /*username*/,
    const std::wstring& /*password*/,
    bool /*passiveFtp*/,
    const std::atomic_bool* /*pBreak*/)
{
    WZResult result;
    result.SetResult(DL_CREATE_SESSION, 0, L"[CurlFileDownloader] Not available in the web build: %ls", url.c_str());
    return result;
}
