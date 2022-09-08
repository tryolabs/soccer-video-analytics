chelsea_filter = {
    "name": "Chelsea",
    "lower_hsv": (112, 0, 0),
    "upper_hsv": (179, 255, 255),
}

chelsea_gk_filter = {
    "name": "Chelsea",
    "lower_hsv": (46, 88, 0),
    "upper_hsv": (55, 255, 255),
}

city_filter = {
    "name": "Man City",
    "lower_hsv": (91, 0, 0),
    "upper_hsv": (112, 255, 255),
}

referee_filter = {
    "name": "Referee",
    "lower_hsv": (0, 0, 0),
    "upper_hsv": (179, 255, 51),
}

filters = [
    chelsea_filter,
    chelsea_gk_filter,
    city_filter,
    referee_filter,
]
