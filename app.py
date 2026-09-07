import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from pathlib import Path

from utils.constants import (
    PREDICTION_THRESHOLD,
    REQUIRED_COLUMNS,
)
from utils.descriptions import FEATURE_DESCRIPTIONS
from utils.feature_engineering import engineer_features
from utils.load_resources import (
    load_resources,
    load_shap_explainer,
)
from utils.shap_utils import create_local_shap_explanation


# ===================================================
# PAGE SETUP
# ===================================================

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Credit Card Fraud Detector")

st.write(
    "Upload a CSV containing one transaction. "
    "Extra CSV columns will be ignored."
)

st.markdown(
    "**Don't have a formatted CSV?** Download the sample "
    "transaction CSV file below and upload it to test the app. "
    "It contains one example transaction with all required columns."
)

sample_csv_path = (
    Path(__file__).resolve().parent
    / "sample_transaction.csv"
)

st.download_button(
    label="Download Sample Transaction CSV File",
    data=sample_csv_path.read_bytes(),
    file_name="sample_transaction.csv",
    mime="text/csv",
)


# ===================================================
# LOAD MODEL AND DICTIONARIES
# ===================================================

try:
    (
        model,
        model_features,
        card_database,
        category_dict,
    ) = load_resources()

except FileNotFoundError as error:
    st.error(
        f"Required file was not found: {error}"
    )
    st.stop()

except Exception as error:
    st.error(
        "Could not load the saved files: "
        f"{type(error).__name__}: {error}"
    )
    st.stop()


# ===================================================
# LOAD SHAP EXPLAINER
# ===================================================

try:
    shap_explainer = load_shap_explainer(
        model
    )

except Exception as error:
    shap_explainer = None

    st.warning(
        "The model loaded, but the SHAP explainer "
        f"could not be created: {error}"
    )


# ===================================================
# CSV UPLOAD
# ===================================================

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)


if uploaded_file is not None:
    try:
        full_df = pd.read_csv(
            uploaded_file,
            dtype={"cc_num": str},
        )

        full_df.columns = (
            full_df.columns
            .str.strip()
        )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in full_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if len(full_df) != 1:
            st.error(
                "Upload a CSV containing exactly "
                "one transaction row."
            )
            st.stop()

        input_df = full_df[
            REQUIRED_COLUMNS
        ].copy()

        st.subheader(
            "Transaction input"
        )

        st.dataframe(
            input_df,
            width="stretch",
            hide_index=True,
        )

        if st.button(
            "Predict fraud",
            type="primary",
        ):
            transaction = input_df.iloc[0]

            model_input = engineer_features(
                transaction,
                model_features,
                card_database,
                category_dict,
            )

        
            fraud_probability = float(
                model.predict_proba(
                    model_input
                )[0, 1]
            )

            prediction = int(
                fraud_probability
                >= PREDICTION_THRESHOLD
            )

            # ---------------------------------------
            # Prediction result
            # ---------------------------------------

            st.subheader(
                "Prediction result"
            )

            if prediction == 1:
                st.error(
                    "Potential fraud detected"
                )

                st.write(
                    "Recommended action: request "
                    "verification or temporarily block "
                    "the transaction."
                )

            else:
                st.success(
                    "Transaction predicted as legitimate"
                )

                st.write(
                    "Recommended action: approve "
                    "the transaction."
                )

            # ---------------------------------------
            # SHAP explanation
            # ---------------------------------------

            st.subheader(
                "Why the model made this prediction"
            )

            st.write(
                "The chart shows how each feature "
                "affected the model's prediction for "
                "this specific transaction."
            )

            try:
                local_explanation = (
                    create_local_shap_explanation(
                        model_input,
                        shap_explainer,
                    )
                )

                plt.figure(
                    figsize=(10, 6)
                )

                shap.plots.waterfall(
                    local_explanation,
                    max_display=len(model_features),
                    show=False,
                )

                shap_figure = plt.gcf()

                plt.tight_layout()

                st.pyplot(
                    shap_figure,
                    width="stretch",
                )

                plt.close(
                    shap_figure
                )

                st.info(
                    "Features shown in red push the "
                    "prediction toward fraud. Features "
                    "shown in blue push the prediction "
                    "toward legitimate."
                )

            except Exception as shap_error:
                st.warning(
                    "The prediction succeeded, but the "
                    "SHAP explanation could not be shown: "
                    f"{type(shap_error).__name__}: "
                    f"{shap_error}"
                )

            # ---------------------------------------
            # Engineered features
            # ---------------------------------------

            with st.expander(
                "View engineered model features"
            ):
                st.dataframe(
                    model_input,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("---")

                st.subheader(
                    "Feature descriptions"
                )

                for feature in model_features:
                    description = (
                        FEATURE_DESCRIPTIONS.get(
                            feature,
                            "No description available.",
                        )
                    )

                    st.markdown(
                        f"**{feature}** — {description}"
                    )

    except ValueError as error:
        st.error(
            f"Invalid transaction data: {error}"
        )

    except Exception as error:
        st.error(
            "Prediction failed: "
            f"{type(error).__name__}: {error}"
        )
