#pragma once

#include "UI/Core/WindowObject.h"
#include "UI/Core/WindowManager.h"
#include "UI/Combat/CastleSiegeScoreboardRmlModel.h"
#include "UI/RmlBridge/RmlThemedView.h"

#include <cstdint>

namespace Rml
{
class ElementDocument;
}

namespace mu::ui::window
{
// The point mode Castle Siege scoreboard on the siege map: the guilds' points, the castle holder,
// the time left and the next defense tick. castle_siege_scoreboard.rml draws it; C++ feeds it from
// GameLogic::CastleSiege::g_Scoreboard and hides it off the siege map and after the final scores.
class CCastleSiegeScoreboard : public CObject
{
public:
    CCastleSiegeScoreboard() = default;
    ~CCastleSiegeScoreboard() override;

    bool Create(CManager* manager);
    void Release();

    bool UpdateMouseEvent() override { return true; }
    bool UpdateKeyEvent() override { return true; }
    bool Update() override;
    bool Render() override { return true; }
    Rml::ElementDocument* GetPlacedDocument() const override { return m_RmlView.Document(); }
    float GetLayerDepth() override;

private:
    bool ShouldShow(std::uint32_t nowMs) const;
    void SyncRmlModel(std::uint32_t nowMs);
    void SyncLines();
    void SyncTimers(std::uint32_t nowMs);

    void BindRmlModel(Rml::DataModelConstructor& c, CastleSiegeScoreboardRmlModel& model);

    CManager* m_manager = nullptr;
    std::uint32_t m_syncedVersion = 0;
    UI::RmlBridge::ThemedView<CastleSiegeScoreboardRmlModel> m_RmlView{"castle_siege_scoreboard",
        [this](Rml::DataModelConstructor& c, CastleSiegeScoreboardRmlModel& model) { BindRmlModel(c, model); },
        {{"Data/Interface/RmlUi/castle_siege_scoreboard.rml"}}};
};
} // namespace mu::ui::window
