// Tests of the skill hotkey queue.
//
// Hotkeys pressed during a swing are cast in press order, so a fast 1-2-3 combo macro casts 1-2-3
// instead of whichever skill happens to be selected when a swing ends.

#include "doctest.h"

#include "GameLogic/Combat/SkillInputQueue.h"

using GameLogic::Combat::SkillInputQueue;

TEST_CASE("Skills are cast in press order")
{
    SkillInputQueue queue;
    queue.Push(1, 0);
    queue.Push(2, 10);
    queue.Push(3, 20);

    CHECK(queue.Peek(30) == 1);
    queue.Pop();
    CHECK(queue.Peek(30) == 2);
    queue.Pop();
    CHECK(queue.Peek(30) == 3);
    queue.Pop();
    CHECK(queue.Peek(30) == -1);
    CHECK(queue.Empty());
}

TEST_CASE("Repeated presses of the same skill don't stack")
{
    SkillInputQueue queue;
    queue.Push(2, 0);
    queue.Push(2, 10);
    queue.Pop();

    CHECK(queue.Empty());
}

TEST_CASE("A full queue ignores further presses instead of skipping a step")
{
    SkillInputQueue queue;
    queue.Push(1, 0);
    queue.Push(2, 0);
    queue.Push(3, 0);
    queue.Push(1, 0);

    CHECK(queue.Peek(0) == 1);
    queue.Pop();
    CHECK(queue.Peek(0) == 2);
    queue.Pop();
    CHECK(queue.Peek(0) == 3);
    queue.Pop();
    CHECK(queue.Empty());
}

TEST_CASE("Presses which weren't cast in time expire")
{
    SkillInputQueue queue;
    queue.Push(1, 0);
    queue.Push(2, 900);

    CHECK(queue.Peek(SkillInputQueue::kMaximumAgeMs + 1) == 2);
    CHECK(queue.Peek(900 + SkillInputQueue::kMaximumAgeMs + 1) == -1);
}

TEST_CASE("Clear forgets all presses")
{
    SkillInputQueue queue;
    queue.Push(1, 0);
    queue.Push(2, 0);
    queue.Clear();

    CHECK(queue.Empty());
    CHECK(queue.Peek(0) == -1);
}
