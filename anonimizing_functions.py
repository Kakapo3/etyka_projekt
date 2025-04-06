import pandas as pd
import numpy as np


def add_noise(df, column, max_noise=5):
    """
    Dodaje zakłócenia do wartości liczbowych w kolumnie, zmieniając je o +/- max_noise (losowo).
    """
    noisy_df = df.copy()
    if column in noisy_df.columns:
        noise = np.random.randint(-max_noise, max_noise + 1, size=len(noisy_df))
        noisy_df[column] = noisy_df[column] + noise
    return noisy_df


def generalize(df, column, percentage=10):
    """
    Generalizuje wartości liczbowe w kolumnie przez zamianę ich na przedziały.
    Przedziały mają wielkość opartą na podanym procencie z maksymalnej wartości w kolumnie.
    """
    generalized_df = df.copy()
    if column in generalized_df.columns:
        col_min = generalized_df[column].min()
        col_max = generalized_df[column].max()
        range_size = (col_max - col_min) * (percentage / 100)

        def to_interval(val):
            lower = (val // range_size) * range_size
            upper = lower + range_size
            return f"[{int(lower)}, {int(upper)})"

        generalized_df[column] = generalized_df[column].apply(to_interval)
    return generalized_df


def suppress_column(df, column):
    """
    Usuwa (supresja) daną kolumnę z DataFrame.
    """
    suppressed_df = df.copy()
    if column in suppressed_df.columns:
        suppressed_df = suppressed_df.drop(columns=[column])
    return suppressed_df


def perturb_data(df, column, std_dev=1.0):
    """
    Wprowadza niewielkie zakłócenia do wartości liczbowych poprzez dodanie szumu normalnego (Gaussa).
    """
    perturbed_df = df.copy()
    if column in perturbed_df.columns:
        noise = np.random.normal(loc=0, scale=std_dev, size=len(perturbed_df))
        perturbed_df[column] = perturbed_df[column] + noise
    return perturbed_df
