# SPDX-License-Identifier: Apache-2.0

print('start')

"""
.. _l-example-lightgbm:

Converts a LightGBM model
=========================

This example trains a `LightGBM
<https://lightgbm.readthedocs.io/en/latest/>`_
model on the Iris datasets and converts it
into ONNX.

.. contents::
    :local:

Train a model
+++++++++++++

"""
import os
import matplotlib.pyplot as plt
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import numpy
import onnx
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm
from lightgbm import LGBMClassifier, Dataset, train as train_lgbm
import onnxruntime as rt
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LGBMClassifier()
clr.fit(X_train, y_train)
print(clr)
native_pred=clr.predict(X_test, pred_leaf=True, pred_contrib=True)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_type = [("float_input", FloatTensorType([None, 4]))]
onx = convert_lightgbm(clr, initial_types=initial_type, decision_path=True)

###################################
# Compute the predictions with onnxruntime
# ++++++++++++++++++++++++++++++++++++++++

sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
n1 = output_names[0]
n2 = output_names[1]
n3 = output_names[2]
pred_onx = sess.run([n1, n2, n3], {input_name: X_test.astype(numpy.float32)})
print(n1)
print(pred_onx[0])
print(n2)
print(pred_onx[1])
print(n3)
print(pred_onx[2])

model_path = "leaf_pred.onnx"
with open(model_path, "wb") as f:
    f.write(onx.SerializeToString())
