# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_regressor_output_shapes

def calculate_lgbm_linear_regressor_output_shapes(operator):
    # apply the common calculator first
    calculate_linear_regressor_output_shapes(operator)

    # common calculator sets only the primary output's shape
    # now set shapes ofr decision_path, decision_leaf if present
    # TODO: is this actually relevant for regressor?
    N = operator.inputs[0].get_first_dimension()
    for n in range(1, len(operator.outputs)):
        operator.outputs[n].type.shape = [N, 1]

register_shape_calculator("LgbmRegressor", calculate_lgbm_linear_regressor_output_shapes)
