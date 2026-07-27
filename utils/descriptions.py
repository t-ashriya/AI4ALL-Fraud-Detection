FEATURE_DESCRIPTIONS = {
    "amt": (
        "Transaction amount in dollars."
    ),
    "amt_vs_avg": (
        "Ratio of the current transaction amount to "
        "the card's historical average transaction amount."
    ),
    "amt_bin_3": (
        "1 if the transaction amount is between $500 "
        "and $1,000; otherwise 0."
    ),
    "amt_bin_2": (
        "1 if the transaction amount is between $200 "
        "and $500; otherwise 0."
    ),
    "card_avg_amt": (
        "Average amount historically spent using this "
        "credit card."
    ),
    "hour_cos": (
        "Cosine transformation of the transaction hour "
        "to capture time-of-day patterns."
    ),
    "amt_bin_4": (
        "1 if the transaction amount is greater than "
        "$1,000; otherwise 0."
    ),
    "amt_bin_0": (
        "1 if the transaction amount is $50 or less; "
        "otherwise 0."
    ),
    "hour_sin": (
        "Sine transformation of the transaction hour "
        "to capture cyclic daily patterns."
    ),
    "amt_bin_1": (
        "1 if the transaction amount is between $50 "
        "and $200; otherwise 0."
    ),
    "trans_count_24h": (
        "Number of transactions made with this card "
        "during the previous 24 hours."
    ),
    "category": (
        "Encoded merchant category based on how "
        "frequently that category appears in the "
        "training data."
    ),
    "secs_since_last": (
        "Number of seconds since the previous "
        "transaction made with this card."
    ),
    "is_night": (
        "1 if the transaction occurred between "
        "10 PM and 3 AM; otherwise 0."
    ),
    "age": (
        "Cardholder's age in years."
    ),
    "hour": (
        "Hour of the day when the transaction "
        "occurred (0–23)."
    ),
}