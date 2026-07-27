import numpy as np
import pandas as pd
import shap


# ===================================================
# BUILD LOCAL SHAP EXPLANATION
# ===================================================

def create_local_shap_explanation(
    model_input: pd.DataFrame,
    shap_explainer,
) -> shap.Explanation:
    """Calculate SHAP values for one transaction."""

    if shap_explainer is None:
        raise RuntimeError(
            "The SHAP explainer is unavailable."
        )

    shap_output = shap_explainer(
        model_input
    )

    shap_values = np.asarray(
        shap_output.values
    )

    base_values = np.asarray(
        shap_output.base_values
    )

    # For class dimension.
    if shap_values.ndim == 3:
        local_values = shap_values[
            0,
            :,
            1,
        ]

        if base_values.ndim >= 2:
            local_base_value = float(
                base_values[
                    0,
                    1,
                ]
            )

        else:
            local_base_value = float(
                base_values
                .reshape(-1)[-1]
            )

    else:
        local_values = shap_values[
            0
        ]

        local_base_value = float(
            base_values
            .reshape(-1)[0]
        )

    return shap.Explanation(
        values=local_values,
        base_values=local_base_value,
        data=model_input.iloc[
            0
        ].to_numpy(),
        feature_names=(
            model_input
            .columns
            .tolist()
        ),
    )