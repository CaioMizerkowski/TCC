import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "..")
import lib.tf_levenberg_marquardt.levenberg_marquardt as lm
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from itertools import product
from time import time
from itertools import product


def nmse_error(y, y_hat):
    nmse = 10 * np.log10(np.sum(np.abs(y - y_hat) ** 2) / np.sum(np.abs(y) ** 2))
    return nmse


def load_csv2numpy(p):
    return np.loadtxt(p, delimiter=",", dtype=np.float32, skiprows=1)


def z2n(n):
    return n if n < 0 else None


def prep_data_y(y, M, My=1):
    a0 = np.abs(y)
    end = z2n(-M - 1)
    abs = [
        a0[:end],
    ]
    for m in range(My - 1):
        end = z2n(-M + m)
        abs.append(a0[m + 1 : end])
    abs = np.array(abs).T
    return abs


def prep_data(x, M):
    a0 = np.abs(x)
    abs = []
    for m in range(M):
        abs.append(a0[M + m + 1 - M : m - M])
    abs.append(a0[M + 1 :])
    abs = np.array(abs).T

    p0 = np.angle(x)
    phase = []
    for m in range(M):
        phase.append(p0[M + m + 1 - M : m - M])
    phase.append(p0[M + 1 :])
    phase = np.array(phase).T

    delta = []
    for m in range(1, phase.shape[1]):
        delta.append(phase[:, m] - phase[:, m - 1])
    delta = np.array(delta).T

    d_cos = np.cos(delta)
    d_sin = np.sin(delta)

    inputs = abs
    if M > 0:
        inputs = np.append(inputs, d_cos, axis=1)
        inputs = np.append(inputs, d_sin, axis=1)
    return inputs


def recuperar_data(s, x):
    y = s * np.exp(1j * np.angle(x))
    return y


def gerar_s(y, x):
    s = y * np.exp(-1j * np.angle(x))
    return s.real, s.imag


def normalize_data(x):
    return (x) / np.std(x)


def gen_model(M, HL, name, activation="tanh"):
    size_input = 1 + 3 * (M)
    inputs = keras.Input(shape=(size_input,))
    hidden = layers.Dense(HL, activation=activation)(inputs)
    output = layers.Dense(1, activation="linear")(hidden)
    model = keras.Model(inputs=inputs, outputs=output, name=name)
    model = lm.ModelWrapper(model)
    model.call = tf.function(model.call)
    return model


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=lm.MeanSquaredError(),
        metrics=["accuracy", "mse"],
        run_eargly=False,
    )


def fit_model(model, epochs, inputs, s):
    history = model.fit(inputs, s, epochs=epochs, batch_size=128, verbose=0)
    return history


def run_model(x, model, M):
    inputs = prep_data(x, M)
    # Modificado por questões de performance, o model.predict acaba ficando sem o ganho de usar o tf.function no modelo
    s_hat_real = (
        model[0](inputs)
        .numpy()
        .reshape(
            -1,
        )
    )
    s_hat_imag = (
        model[1](inputs)
        .numpy()
        .reshape(
            -1,
        )
    )

    s_hat = s_hat_real + 1j * s_hat_imag
    y_hat = recuperar_data(s_hat, x[M + 1 :])
    return y_hat


def validation(model, x, y, M):
    y_hat = run_model(x, model, M)
    nmse = nmse_error(y[M + 1 :], y_hat)
    return nmse, y_hat


def find_best_M_HL(x, y, M_min=0, M_max=10, HL_min=1, HL_max=10):
    best_M = 0
    best_model = None
    best_nmse = np.inf
    best_y_hat = None
    metaparams = product(range(M_min, M_max + 1), range(HL_min, HL_max + 1))
    for M, HL in metaparams:
        start = time()
        inputs = prep_data(x, M)
        s_real, s_imag = gerar_s(y[M + 1 :], x[M + 1 :])

        model_real = gen_model(M, HL, "parte_real")
        model_imag = gen_model(M, HL, "parte_imag")

        compile_model(model_real)
        compile_model(model_imag)
        his = [0, 0]
        his[0] = fit_model(model_real, 1000, inputs, s_real)
        his[1] = fit_model(model_imag, 1000, inputs, s_imag)

        model = (model_real, model_imag)
        # nmse = validation_models(model_real, model_imag, inputs, x, y)
        nmse, y_hat = validation(model, x, y, M)
        end = time()
        if nmse < best_nmse:
            best_nmse = nmse
            best_model = model
            best_M = M
            best_y_hat = y_hat
            print("*** BEST ***")
            print(f"Shape:{inputs.shape[1]} M:{M} HL:{HL} nmse:{nmse} time:{end-start}")
    return best_model, best_M, best_y_hat, best_nmse, his
