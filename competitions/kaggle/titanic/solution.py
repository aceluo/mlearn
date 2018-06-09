# coding: utf-8
import pandas as pd
import numpy as np
"""
Data columns: 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

Step1: transform some of the data.

1. separate the `Age` in to groups by a range of 5 years.
    0 ~ 4:   1
    5 ~ 9: 2
    10 ~ 14: 3,
    15 ~ 19: 4
    ...

    for people with no age info:
     1) SibSp is 0 and Parch is 0, set to 6
     2) SibSp is 0 and Parch > 0, set to 6
     3) SibSp > 0 and Parch > 0, set to 6


2. parse Cabin information and mapped to labels to the first character of their cabin number:
    A16, A20 ->  A
    if multiple characters, just concat to one string and remove duplicates
    F G73  -> FG
    B12 B13 B14 -> B
    if blank, use NA
3. ticket number is removed as this should not has strong relationship with survived info.
4. discard Name info.

Step2:  check correlation of each column and see if there's anything to remove

Step3:  split the train set into 3 parts: 60% actual training, 20% verification, 20% cross validation.

"""


def transform_age(df, column_name='Age'):
    # assert isinstance(df, pd.DataFrame)
    for idx, value in enumerate(df[column_name]):
        new_val = int(value)/5 + 1
        if value == 0:
            new_val = 6
        df[column_name][idx] = new_val


def transform_cabin(df, column_name='Cabin'):
    # assert isinstance(df, pd.DataFrame)
    number_list = [str(c) for c in range(10)]
    for idx, value in enumerate(df[column_name]):
        new_val = str()
        for c in list(str(value)):
            print(c)
            if len(c.strip()) > 0 and c not in number_list and c not in list(new_val):
                new_val += str(c)
        df[column_name][idx] = new_val


if __name__ == '__main__':
    l = {'Cabin': ['B123', 'B12 B34', 'F G12']}
    transform_cabin(l)
    pass
