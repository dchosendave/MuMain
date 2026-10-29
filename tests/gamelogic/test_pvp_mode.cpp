#include <doctest.h>

#include "GameLogic/Combat/PvpMode.h"

using GameLogic::Combat::CtrlDoubleTap;

namespace
{
    // A clean tap: Ctrl down at `at`, up 80 ms later. Returns whether it completed a double tap.
    bool Tap(CtrlDoubleTap& taps, std::uint64_t at)
    {
        taps.OnCtrlDown(at);
        return taps.OnCtrlUp(at + 80);
    }
}

TEST_CASE("Two quick Ctrl taps toggle PvP mode")
{
    CtrlDoubleTap taps;
    CHECK_FALSE(Tap(taps, 1000));
    CHECK(Tap(taps, 1250));
}

TEST_CASE("A third tap starts a new double tap")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    CHECK(Tap(taps, 1250));
    CHECK_FALSE(Tap(taps, 1500));
    CHECK(Tap(taps, 1750));
}

TEST_CASE("Taps too far apart don't count")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    CHECK_FALSE(Tap(taps, 1000 + 80 + CtrlDoubleTap::kMaxGapMs + 1));
}

TEST_CASE("Holding Ctrl isn't a tap")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    taps.OnCtrlDown(1200);
    CHECK_FALSE(taps.OnCtrlUp(1200 + CtrlDoubleTap::kMaxTapMs + 1));
}

TEST_CASE("Ctrl + another key isn't a tap")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    taps.OnCtrlDown(1200);
    taps.OnOtherInput(); // e.g. Ctrl + 1 to save a skill hotkey
    CHECK_FALSE(taps.OnCtrlUp(1280));
    CHECK_FALSE(Tap(taps, 1400));
    CHECK(Tap(taps, 1600));
}

TEST_CASE("Another key between the taps breaks the double tap")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    taps.OnOtherInput();
    CHECK_FALSE(Tap(taps, 1200));
}

TEST_CASE("Key repeat while Ctrl is held is ignored")
{
    CtrlDoubleTap taps;
    Tap(taps, 1000);
    taps.OnCtrlDown(1200);
    taps.OnCtrlDown(1230);
    CHECK(taps.OnCtrlUp(1280));
}
