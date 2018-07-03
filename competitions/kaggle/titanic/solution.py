# coding: utf-8
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

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


class TitanicDataPreProcessor(object):
    # columns_to_use = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked']

    def __init__(self):
        self._df = None
        self._label_encoders = None

    @property
    def df(self):
        return self._df

    def _create_label_encoders(self, column_list):
        self._label_encoders = {}
        for column_name in column_list:
            self.df.loc[self.df[column_name].isnull(), column_name] = 'NA'
            self._label_encoders.update({column_name:
                                             LabelEncoder().fit(list(set(self.df[column_name])))})

    def inverse_transform(self, column_name, values):
        if not isinstance(self._label_encoders, dict):
            raise Exception('Process data firstly')
        label_encoder = self._label_encoders.get(column_name, None)
        if not isinstance(label_encoder, LabelEncoder):
            raise Exception('Invalid label encoder type: {encoder_type}'.format(encoder_type=label_encoder.__class__))
        return label_encoder.inverse_transform(values)

    def _build_cabin_type(self):
        def transform_cabin_value(value):
            new_val = str()

            for c in list(str(value)):
                if len(c.strip()) > 0 and c not in number_list and c not in list(new_val):
                    new_val += str(c)
            if pd.isna(value):
                new_val = 'NA'
            return new_val

        column_name = 'CabinType'
        self._df[column_name] = self._df['Cabin'].apply(transform_cabin_value)
        return column_name

    def _build_age_group(self):

        column_name = 'AgeGroup'
        self._df.loc[self._df['Age'].isnull(), 'Age'] = self._df['Age'].mean()
        self._df[column_name] = pd.cut(self._df['Age'], bins=range(0, 100, 4), labels=False, include_lowest=True)

        return column_name

    def _build_fare_group(self):
        column_name = 'FareGroup'
        fare_mean = self._df['Fare'].mean()
        self._df.loc[self._df['Fare'].isnull(), 'Fare'] = fare_mean
        self._df[column_name] = pd.cut(self._df['Fare'], bins=range(0, 2000, 10), include_lowest=True,
                                       labels=False)
        return column_name

    def process(self, src_csv):
        """
        main steps to preprocess data.
        :return: dataframe after being preprocessed.
        """
        df = pd.read_csv(src_csv, header=0)

        self._df = df

        columns_to_encode = ['Sex', 'Embarked']
        age_group_name = self._build_age_group()
        cabin_type_name = self._build_cabin_type()
        fare_group_name = self._build_fare_group()
        columns_to_encode += [cabin_type_name]
        self._create_label_encoders(column_list=columns_to_encode)

        for column_name, label_encoder in self._label_encoders.iteritems():
            print('encode: {column_name}'.format(column_name=column_name))
            df[column_name] = label_encoder.transform(df[column_name])

        self._label_encoders = None

        return self._df


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics

    # src_csv_list = ('/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/train/train.csv',
    #                 '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/test/test.csv',
    #                 )
    # for src_csv in src_csv_list:
    #     preprocess_data_csv(src_csv)

    train_csv = '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/train/train.csv'

    test_csv = '/Users/ace_luo/Developers/mlearn/competitions/kaggle/titanic/data/test/test.csv'

    pre_processor = TitanicDataPreProcessor()

    df_train = pre_processor.process(train_csv)
    df_test = pre_processor.process(test_csv)

    un_used_columns = ['Name', 'Ticket', 'PassengerId']

    df_result = pd.DataFrame()
    df_result['PassengerId'] = df_test['PassengerId']

    df_test = df_test[df_test.columns.drop(un_used_columns + ['Age', 'Cabin', 'Fare'])]

    X = df_train[df_train.columns.drop(un_used_columns + ['Survived', 'Age', 'Cabin', 'Fare'])]
    y = df_train['Survived']

    # check covariance of features with 'Survived'
    X1 = df_train[df_train.columns.drop(un_used_columns + ['Age', 'Cabin'])]
    for column_name in X1.columns:
        print('%s ---> cov: %s' % (column_name, X1['Survived'].corr(X1[column_name])))

    # find parameter C
    for c in range(1, 10, 1):
        C = c/10.0
        clf = svm.SVC(C=C)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
        print('c = {C}, f1_mean = {f1_mean}, f1_std = {f1_std}'.format(C=C,
                                                                       f1_mean=cv_scores.mean(),
                                                                       f1_std=cv_scores.std()))
    C = 0.9
    svm_clf = svm.SVC(C=C)

    svm_clf.fit(X, y)

    train_predictions = svm_clf.predict(X)
    print(metrics.classification_report(y, train_predictions))

    test_predictions = svm_clf.predict(df_test)


    df_result['Survived'] = test_predictions

    df_result.to_csv('prediction_C_%s.csv' % C, index=False, columns=['PassengerId', 'Survived'])

    # pipeline = Pipeline()

    # random_forest = RandomForestClassifier()

    # random_forest.fit(X=X, y=y)
    #
    # prediction = random_forest.predict(df_test)
    #
    # print(prediction)
