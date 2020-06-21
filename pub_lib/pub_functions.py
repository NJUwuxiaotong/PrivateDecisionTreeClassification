import math
import random

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np


def get_Laplace_PDF(value):
    l = 1
    if value < 0:
        return math.exp(value/l)/2
    else:
        return 1 - math.exp(-1*value/l)/2


def get_probability_from_Laplace(noisy_magnitude, value):
    return math.exp(-1*math.fabs(value)/noisy_magnitude)/(2*noisy_magnitude)


def generate_random_value_from_Laplace(start_v, stop_v, noisy_magnitude):
    while True:
        value = random.uniform(start_v, stop_v)
        random_pr = random.uniform(0, 1)
        if random_pr < get_probability_from_Laplace(noisy_magnitude, value):
            return value


def generate_random_value_from_exponential(pr_values):
    """
    pr_values: a list of values
    """
    values = np.array(pr_values)
    values = np.exp(values)
    values = values/np.sum(values)
    return generate_random_from_specified_pr(values)


def generate_random_from_specified_pr(pr_values):
    """
    pr_values: np.array
    """
    total_sum = np.sum(pr_values)
    if total_sum == 0:
        print("Error: The sum of probabilities is equal to 0.")
        exit(1)

    pr_values = pr_values/total_sum
    chosen_index = 0
    sum_pr = 0
    pr = random.random()
    for sub_pr in pr_values:
        sum_pr += sub_pr
        if pr <= sum_pr:
            return chosen_index
        chosen_index += 1


def show_pdf(pdf_type):
    if pdf_type not in ["Laplace"]:
        print("It only supports Laplace.")
        return

    interate_num = 100000
    result = list()
    for i in range(interate_num):
        if pdf_type == "Laplace":
            result.append(generate_random_value_from_Laplace(-100, 100, 2))

    sns.distplot(result, bins=np.arange(-100, 100, 0.01))
    plt.show()
