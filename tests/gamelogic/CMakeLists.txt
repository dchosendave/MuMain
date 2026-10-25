mu_add_test(
    NAME test_boid_steering
    SOURCES
        test_boid_steering.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_head_movement
    SOURCES
        test_head_movement.cpp
        ${CMAKE_SOURCE_DIR}/src/source/App/Platform/Windows/Winmain.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_earthquake_falloff
    SOURCES
        test_earthquake_falloff.cpp
        ${CMAKE_SOURCE_DIR}/src/source/App/Platform/Windows/Winmain.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_reference_frame_steps
    SOURCES
        test_reference_frame_steps.cpp
        ${CMAKE_SOURCE_DIR}/src/source/App/Platform/Windows/Winmain.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_direction_turning
    SOURCES
        test_direction_turning.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_personal_shop_currency
    SOURCES
        test_personal_shop_currency.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_swing_action
    SOURCES
        test_swing_action.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_attack_speed_curve
    SOURCES
        test_attack_speed_curve.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_skill_input_queue
    SOURCES
        test_skill_input_queue.cpp
    LINK_LIBS
        MuClient
)

mu_add_test(
    NAME test_classic_swing
    SOURCES
        test_classic_swing.cpp
    LINK_LIBS
        MuClient
)
