# 기준 xgboost
# 1. FI 0 제거 또는 하위 30% 제거
# 2. 디폴트와 성능 비교

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
