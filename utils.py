# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 08:28:25 2023

@author: pathouli
"""


def clean_txt(str_in):
    import re
    clean_text_t = re.sub("[^A-Za-z']+", " ", str_in).strip().lower()  # strip removes the trailing space
    return clean_text_t


def wrd_freq(str_in):
    # use for LARGE corpuses
    import collections
    word_freq = collections.Counter(str_in.split())
    print(dict(word_freq))
    return word_freq


def wrd_freq_pd(str_in):
    import pandas as pd
    my_pd_new = pd.DataFrame()
    token_t = str_in.split()
    for word in set(token_t):
        tmp = pd.DataFrame({'word': word,
                            'count': token_t.count(word)}, index=[0])
        my_pd_new = pd.concat([my_pd_new, tmp], ignore_index=True)
    return my_pd_new


def file_senti(file_in):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vs_t = SentimentIntensityAnalyzer()
    f_t = open(file_in, 'r')
    txt_t = f_t.read()
    f_t.close()
    txt_t = clean_txt(txt_t)
    senti_t = vs_t.polarity_scores(txt_t)["compound"]
    return senti_t


def file_reader(path_in):
    txt_t = ""
    try:
        f_t = open(path_in, 'r', encoding="utf-8")
        txt_t = f_t.read()
        f_t.close()
        txt_t = clean_txt(txt_t)
    except:
        print("Problem with", f_t)
        pass
    return txt_t


def file_processor(path_in):
    import pandas as pd
    import os
    # root tracks a specific directory path
    # files is a list of ALL files in a specific directory
    my_pd_t = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
        for name in files:
            t_path = root + "/" + name
            tmp = root.split("/")
            label_t = tmp[-1::][0]
            tmp_txt = file_reader(t_path)
            if len(tmp_txt) != 0:
                pd_t = pd.DataFrame(
                    {"body": tmp_txt, "label": label_t}, index=[0])
                my_pd_t = pd.concat([my_pd_t, pd_t], ignore_index=True)
    return my_pd_t


def rem_sw(str_in):
    # remove stopwords, these are tokens/words that are the most commonly
    # used in the English dictonary and do not help us discriminate between
    # different topics
    # install a python package use the pip install <package_name>
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    my_l = list()
    for word in str_in.split():
        if word not in sw:
            my_l.append(word)
    test_p = ' '.join(my_l)
    return test_p


def stem_fun(str_in):
    # stemming removes affixes of tokens
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    test_p = [ps.stem(word) for word in str_in.split()]
    test_p = ' '.join(test_p)
    return test_p


def write_pickle(obj_in, path_in, file_n_in):
    import pickle
    # save off any object to a file
    pickle.dump(obj_in, open(path_in + file_n_in + ".pk", 'wb'))


def read_pickle(path_i, name_i):
    import pickle
    the_data_t = pickle.load(open(path_i + name_i + ".pk", "rb"))
    return the_data_t


def vectorize_data(df_in, path_in, name_in, m_in, n_in):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    if name_in == "vec":
        vec_t = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        vec_t = TfidfVectorizer(ngram_range=(m_in, n_in))
    vec_data_t = pd.DataFrame(vec_t.fit_transform(
        df_in).toarray())  # be careful memory intensive
    col_names_t = vec_t.get_feature_names_out()
    vec_data_t.columns = col_names_t
    write_pickle(vec_t, path_in, name_in)
    return vec_data_t


def chi_fun(df_in, label_in, k_in, path_in, name_in):
    # chi-square feature selection
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(
        feat_sel.fit_transform(df_in, label_in))
    feat_index = feat_sel.get_support(indices=True)
    feature_names = df_in.columns[feat_index]
    dim_data.columns = list(feature_names)
    write_pickle(feat_sel, path_in, name_in)
    return dim_data


def pca_fun(df_in, var_in, path_in, name_in):
    import pandas as pd
    # dimension reduction technique PCA
    # drawback is you lose transparency
    from sklearn.decomposition import PCA
    pca_fun = PCA(n_components=var_in)
    pca_data_t = pd.DataFrame(pca_fun.fit_transform(df_in))
    exp_var = sum(pca_fun.explained_variance_ratio_)
    print("Exp var", exp_var)
    write_pickle(pca_fun, path_in, name_in)
    return pca_data_t


def train_test_split(df_in, label_in, size_in, path_in, name_in):
    # MODELING STAGE#
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    m = RandomForestClassifier()
    # 80/20 - train model with 80% of the data and test on 20% of the data
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=size_in, random_state=42)
    m.fit(X_train, y_train)  # fit the model
    y_pred = m.predict(X_test)  # test
    # model performance metrics
    perf = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    perf.index = ["precision", "recall", "f-score", "None"]
    print(perf)
    feat_imp = pd.DataFrame(m.feature_importances_)
    feat_imp.index = df_in.columns
    feat_imp.columns = ["fi_score"]
    perc_score = feat_imp[feat_imp.fi_score != 0]
    print("The % that has propensity",
          len(perc_score) / len(feat_imp) * 100, "%")
    write_pickle(m, path_in, name_in)
    return m