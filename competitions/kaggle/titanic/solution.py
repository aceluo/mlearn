# coding: utf-8
import pandas as pd
import numpy as np
import os

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
    then map the string to integer from 1 to x by order.
3. ticket number is removed as this should not has strong relationship with survived info.
4. discard Name info.
5. `embarked` -> mapping to numbers from 1 to 4
    'Q', nan, 'S', 'C' ->  1, 2, 3, 4

Step2:  check correlation of each column and see if there's anything to remove

Step3:  split the train set into 3 parts: 60% actual training, 20% verification, 20% cross validation.

"""


def transform_age(value):
    if pd.isna(value):
        new_val = 6
    else:
        new_val = int(value) / 5 + 1

    return new_val


number_list = [str(c) for c in range(10)]
cabin_list = []


def transform_cabin(value):
    # assert isinstance(df, pd.DataFrame)
    new_val = str()

    for c in list(str(value)):
        if len(c.strip()) > 0 and c not in number_list and c not in list(new_val):
            new_val += str(c)
    if pd.isna(value):
        new_val = 'NA'
    if new_val not in cabin_list:
        cabin_list.append(new_val)
        cabin_list.sort()
    return cabin_list.index(new_val)


def transform_embarked(value):
    val_map = {'Q': 1, 'nan': 2, 'S': 3, 'C': 4}
    return val_map.get(value, 5)


def transform_sex(value):
    val_map = {'male': 1, 'female': 2}
    return val_map.get(value, 1)


def preprocess_dataframe(src, dest):
    unused_column_names = (
        'Name', 'Ticket', 'Fare'
    )

    transform_funcs = {
        'Cabin': transform_cabin,
        'Age': transform_age,
        'Embarked': transform_embarked,
        'Sex': transform_sex
    }

    for column_name in src.columns:
        if column_name not in unused_column_names:
            dest[column_name] = src[column_name]

    for column_name, transform_func in transform_funcs.items():
        dest[column_name] = src[column_name].apply(transform_func)


def preprocess_data_csv(src_path):
    src_csv = src_path
    src_folder = os.path.dirname(src_csv)
    src_filename = os.path.basename(src_csv)

    origin_df = pd.read_csv(src_csv)

    processed_df = pd.DataFrame()

    preprocess_dataframe(origin_df, processed_df)

    processed_df.to_csv(os.path.join(src_folder, 'processed_%s' % src_filename))


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier
    # src_csv_list = ('/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/train/train.csv',
    #                 '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/test/test.csv',
    #                 )
    # for src_csv in src_csv_list:
    #     preprocess_data_csv(src_csv)

    train_csv = '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/train/processed_train.csv'

    test_csv = '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/test/processed_test.csv'

    train_pd = pd.read_csv(train_csv)
    test_pd = pd.read_csv(test_csv)

    X = train_pd





