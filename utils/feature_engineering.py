import numpy as np
import pandas as pd


# ===================================================
# FEATURE ENGINEERING
# ===================================================

def engineer_features(
    transaction: pd.Series,
    model_features,
    card_database,
    category_dict,
) -> pd.DataFrame:
    """
    Convert one raw transaction into the exact
    features expected by the trained XGBoost model.
    """

    # Raw values
    amt = float(
        transaction["amt"]
    )

    cc_num = int(
        transaction["cc_num"]
    )

    unix_time = int(
        transaction["unix_time"]
    )

    transaction_time = pd.to_datetime(
        transaction["trans_date_trans_time"],
        errors="raise",
    )

    dob = pd.to_datetime(
        transaction["dob"],
        errors="raise",
    )

    category_name = str(
        transaction["category"]
    ).strip()

    if amt < 0:
        raise ValueError(
            "Transaction amount cannot be negative."
        )

    # Time features
    hour = int(
        transaction_time.hour
    )

    hour_sin = float(
        np.sin(
            2 * np.pi * hour / 24
        )
    )

    hour_cos = float(
        np.cos(
            2 * np.pi * hour / 24
        )
    )

    is_night = int(
        hour in [22, 23, 0, 1, 2, 3]
    )

    # Age
    today = pd.Timestamp.today()

    age = int(
        (today - dob).days // 365
    )

    if age < 0:
        raise ValueError(
            "Date of birth cannot be in the future."
        )

    # Category count encoding
    if category_name not in category_dict:
        supported_categories = ", ".join(
            sorted(
                str(category)
                for category
                in category_dict.keys()
            )
        )

        raise ValueError(
            f"Unknown category: '{category_name}'. "
            "Supported categories are: "
            f"{supported_categories}"
        )

    category_encoded = int(
        category_dict[
            category_name
        ]
    )

    # Card-history lookup
    card_info = card_database.get(
        cc_num
    )

    if card_info is None:
        card_info = card_database.get(
            str(cc_num)
        )

    if card_info is None:
        raise ValueError(
            "This card number was not found in "
            "the saved card database."
        )

    if "avg_amt" not in card_info:
        raise ValueError(
            "The card-history entry is missing "
            "'avg_amt'."
        )

    if "timestamps" not in card_info:
        raise ValueError(
            "The card-history entry is missing "
            "'timestamps'."
        )

    card_avg_amt = float(
        card_info["avg_amt"]
    )

    previous_timestamps = sorted(
        int(timestamp)
        for timestamp
        in card_info["timestamps"]
        if int(timestamp) < unix_time
    )

    # Historical features
    if previous_timestamps:
        last_timestamp = previous_timestamps[
            -1
        ]

        secs_since_last = max(
            unix_time - last_timestamp,
            0,
        )

        beginning_24h = (
            unix_time - 86400
        )

        previous_count_24h = sum(
            beginning_24h
            <= timestamp
            < unix_time
            for timestamp
            in previous_timestamps
        )

        trans_count_24h = (
            previous_count_24h + 1
        )

    else:
        secs_since_last = 86400
        trans_count_24h = 1

    if card_avg_amt > 0:
        amt_vs_avg = (
            amt / card_avg_amt
        )
    else:
        amt_vs_avg = 1.0

    # Amount bins
    amt_bin_0 = int(
        0 <= amt <= 50
    )

    amt_bin_1 = int(
        50 < amt <= 200
    )

    amt_bin_2 = int(
        200 < amt <= 500
    )

    amt_bin_3 = int(
        500 < amt <= 1000
    )

    amt_bin_4 = int(
        amt > 1000
    )

    feature_values = {
        "amt": amt,
        "amt_vs_avg": amt_vs_avg,
        "amt_bin_0": amt_bin_0,
        "amt_bin_1": amt_bin_1,
        "amt_bin_2": amt_bin_2,
        "amt_bin_3": amt_bin_3,
        "amt_bin_4": amt_bin_4,
        "hour": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_night": is_night,
        "trans_count_24h": trans_count_24h,
        "card_avg_amt": card_avg_amt,
        "category": category_encoded,
        "secs_since_last": secs_since_last,
        "age": age,
    }

    model_input = pd.DataFrame(
        [feature_values]
    )

    missing_features = [
        feature
        for feature in model_features
        if feature not in model_input.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature engineering did not create: "
            + ", ".join(
                missing_features
            )
        )

    return model_input[
        model_features
    ].copy()