window.APP_CONFIG = {
    "grid": {
        "rows": 10,
        "cols": 20,
        "cell": 30
    },
    "trial": {
        "max_steps": 100
    },
    "battery": {
        "max": 100,
        "drain_per_step": 1.0,
        "drain_collision_penalty": 2.0
    },
    "blocks": 1,
    "trials_per_block": 5,
    "manipulations": {
        "volatility": [
            "high",
            "high"
        ],
        "urgency": [
            "off",
            "on"
        ],
        "sensor_noise": [
            "high",
            "high"
        ],
        "map_density": [
            "sparse",
            "sparse"
        ]
    },
    "payoffs": {
        "goal": 100,
        "time_cost": -1,
        "collision_penalty": -40,
        "near_miss_penalty": -10,
        "urgency_bonus": 40
    },
    "drift_values": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
    ],
    "seed": 42,
    "timeout": {
        "trial_wallclock_sec": 90,
        "idle_sec": 30,
        "tour_auto_advance_sec": 10,
        "practice_wallclock_sec": 10,
        "practice_idle_sec": 10,
        "survey_wallclock_sec": 10,
        "survey_idle_sec": 10
    }
};
