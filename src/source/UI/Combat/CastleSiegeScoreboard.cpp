#include "stdafx.h"
#include "UI/Combat/CastleSiegeScoreboard.h"

#include "GameLogic/Events/CastleSiegeScoreboard.h"
#include "UI/RmlBridge/RmlDocumentVisibility.h"
#include "UI/RmlBridge/RmlNativeTextSize.h"
#include "World/MapInfra/MapManager.h"

#include <RmlUi/Core/ElementDocument.h>

#include <cstdio>
#include <string>

using namespace mu::ui::window;
using GameLogic::CastleSiege::ScoreboardState;

namespace
{
constexpr int kSecondsPerMinute = 60;

// Layer depth 1.8, like the other event HUDs: under the windows (RmlStackingOrder.cpp).
constexpr float kLayerDepth = 1.8f;

template <typename Model, typename T>
void Sync(RmlModelBinder<Model>& binder, T Model::* field, const char* name, T value)
{
    Model& model = binder.GetModel();
    if (model.*field == value)
        return;
    model.*field = std::move(value);
    binder.MarkDirty(name);
}

// "41:12"; empty for a negative time, which means "none".
Rml::String MinutesAndSeconds(int seconds)
{
    if (seconds < 0)
        return {};
    char text[16];
    std::snprintf(text, sizeof(text), "%02d:%02d", seconds / kSecondsPerMinute, seconds % kSecondsPerMinute);
    return text;
}
} // namespace

CCastleSiegeScoreboard::~CCastleSiegeScoreboard()
{
    Release();
}

bool CCastleSiegeScoreboard::Create(CManager* manager)
{
    if (manager == nullptr)
        return false;

    m_manager = manager;
    m_manager->AddUIObj(INTERFACE_CASTLE_SIEGE_SCOREBOARD, this);
    m_RmlView.Ensure();
    Show(false);
    return true;
}

void CCastleSiegeScoreboard::Release()
{
    if (m_manager)
    {
        m_manager->RemoveUIObj(this);
        m_manager = nullptr;
    }

    m_RmlView.Release();
}

bool CCastleSiegeScoreboard::Update()
{
    const std::uint32_t nowMs = GetTickCount();
    const bool shouldShow = ShouldShow(nowMs);
    if (shouldShow != IsVisible())
        Show(shouldShow);

    SyncRmlModel(nowMs);
    return true;
}

float CCastleSiegeScoreboard::GetLayerDepth()
{
    return kLayerDepth;
}

bool CCastleSiegeScoreboard::ShouldShow(std::uint32_t nowMs) const
{
    const auto& scoreboard = GameLogic::CastleSiege::g_Scoreboard;
    return !scoreboard.IsExpired(nowMs) && gMapManager.InBattleCastle();
}

void CCastleSiegeScoreboard::SyncRmlModel(std::uint32_t nowMs)
{
    m_RmlView.Ensure();
    if (!m_RmlView.Document())
        return;

    UI::RmlBridge::SyncDocumentVisibilityBehind(m_RmlView.Document(), IsVisible());
    if (!IsVisible())
        return;

    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::boldTextPx, "bold_text_px",
         UI::RmlBridge::NativeTextPx(UI::Scaling::FontRole::Bold));
    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::textPx, "text_px",
         UI::RmlBridge::NativeTextPx(UI::Scaling::FontRole::Normal));
    SyncLines();
    SyncTimers(nowMs);
}

void CCastleSiegeScoreboard::SyncLines()
{
    const auto& scoreboard = GameLogic::CastleSiege::g_Scoreboard;
    if (scoreboard.Version() == m_syncedVersion)
        return;
    m_syncedVersion = scoreboard.Version();

    std::vector<CastleSiegeScoreboardLine> lines;
    Rml::String ownGuild;
    int rank = 0;
    for (const auto& guild : scoreboard.Guilds())
    {
        ++rank;
        lines.push_back({std::to_string(rank), guild.name, std::to_string(guild.points), guild.holdsCastle, guild.isOwnGuild});
        if (guild.isOwnGuild)
            ownGuild = guild.name + " #" + std::to_string(rank);
    }

    auto& model = m_RmlView.GetModel();
    model.lines = std::move(lines);
    m_RmlView.MarkDirty("lines");
    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::ownGuild, "own_guild", ownGuild);
    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::final, "final", scoreboard.State() == ScoreboardState::Final);
}

void CCastleSiegeScoreboard::SyncTimers(std::uint32_t nowMs)
{
    const auto& scoreboard = GameLogic::CastleSiege::g_Scoreboard;
    const bool running = scoreboard.State() == ScoreboardState::Running;
    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::timeLeft, "time_left",
         running ? MinutesAndSeconds(scoreboard.RemainingSeconds(nowMs)) : Rml::String());
    Sync(m_RmlView.Binder(), &CastleSiegeScoreboardRmlModel::nextDefenseTick, "next_defense_tick",
         running ? MinutesAndSeconds(scoreboard.NextDefenseTickSeconds(nowMs)) : Rml::String());
}

void CCastleSiegeScoreboard::BindRmlModel(Rml::DataModelConstructor& c, CastleSiegeScoreboardRmlModel& model)
{
    c.Bind("bold_text_px", &model.boldTextPx);
    c.Bind("text_px", &model.textPx);
    c.Bind("final", &model.final);
    c.Bind("time_left", &model.timeLeft);
    c.Bind("own_guild", &model.ownGuild);
    c.Bind("next_defense_tick", &model.nextDefenseTick);

    auto line = c.RegisterStruct<CastleSiegeScoreboardLine>();
    line.RegisterMember("rank", &CastleSiegeScoreboardLine::rank);
    line.RegisterMember("name", &CastleSiegeScoreboardLine::name);
    line.RegisterMember("points", &CastleSiegeScoreboardLine::points);
    line.RegisterMember("holds", &CastleSiegeScoreboardLine::holdsCastle);
    line.RegisterMember("own", &CastleSiegeScoreboardLine::ownGuild);
    c.RegisterArray<std::vector<CastleSiegeScoreboardLine>>();
    c.Bind("lines", &model.lines);
}
