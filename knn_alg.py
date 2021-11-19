import random

# from test import data
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Dot:

    def __init__(self, cords: list, k_class: int = None):

        self.point = np.array(cords.copy())
        self.k_class = k_class

    def __eq__(self, other: "Dot") -> bool:

        for i in range(len(self.point)):
            if self[i] == other[i]:
                pass
            else:
                return False
        return True

    def __add__(self, other) -> float:

        result = 0
        for i in range(len(self.point)):
            result = result + pow(other.point[i] - self.point[i], 2)
        return np.sqrt(result)

    def __getitem__(self, item: int) -> int:
        return self.point[item]


def find_nearest(test_data: List[Dot], target: Dot, k: int) -> List[int]:
    nearest = []
    nearest_nums = []
    for i in test_data:
        dif = target + i
        if len(nearest) < k:
            nearest.append(i)
            nearest_nums.append(dif)
            continue
        if dif < max(nearest_nums):
            ind = nearest_nums.index(max(nearest_nums))
            nearest_nums[ind] = dif
            nearest[ind] = i
    return [i.k_class for i in nearest]


def convert(data: List[Dot]):
    result = [[] for i in range(len(data[0].point) + 1)]
    for i in data:
        for j in range(len(i.point)):
            result[j].append(i[j])
        result[-1].append(i.k_class)
    return result


def start_knn(test_data: list, unk_data, k):
    for i in range(len(unk_data)):
        temp = find_nearest(test_data, unk_data[i], k)
        count = Counter(temp)
        ind = list(count.values()).index(max(count.values()))
        unk_data[i].k_class = list(count.keys())[ind]
    test_data.extend(unk_data)
    return test_data
