##################################
# Telco Churn Feature Engineering
##################################

# İş Problemi:
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz
# beklenmektedir.

# Veri Seti Hikayesi:
# Veri seti 21 değişken 7043 Gözlem değerine sahiptir.


##################################
# Görev 1: Keşifçi Veri Analizi
##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Adım 1: Genel resmi inceliyiniz.

df = pd.read_csv("week6/Hw6/telco_churn/Telco-Customer-Churn.csv")

df.head()
df.dtypes # TotalCharges object gözüküyor. numerik olmalı.
df.shape  # (7043, 21)
df.isnull().sum()  # nan veri bulunmamaktadır.
df.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# errors= "coerce" ile veride bulunan boşluk değerlerini nan'a çevirdik.
df.dtypes # TotalCharges numerik olarak değiştirildi.


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

df[num_cols].describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


for col in cat_cols:
    cat_summary(df, col)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)

"""
Kategorik değişkenler benzer işlem cat_summary fonksiyonu Adım 3'te yapıldı.
"""

def num_cols_with_target(dataframe, num_cols, target_name):
    for num_col in num_cols:
        print(num_col.center(20, "_"))
        print(pd.DataFrame({"MEAN": dataframe.groupby(target_name)[num_col].mean(),
                            "COUNT": dataframe.groupby(target_name)[num_col].count()}))
        print("\n")


num_cols_with_target(df, num_cols, "Churn")

# Adım 5: Aykırı gözlem analizi yapınız.


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, columns, q1, q3):
    for col_name in columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            print(f"{col_name}  \t True")
        else:
            print(f"{col_name} \t False")


check_outlier(df, num_cols, 0.01, 0.99)
check_outlier(df, num_cols, 0.05, 0.95)
check_outlier(df, num_cols, 0.1, 0.9)

"""
Aykırı değer bulunmamaktadır.
"""

# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()  # NaN değer bulunmamaktadır.

"""
TotalCharges değişkeni içerisinde " " şeklinde boş değerler vardı. Bunları nan'a dönüştürmüştüm. Bu nan değerleri içeren
satırlarda tenure değerlerinin 0 olduğu görülmüktedir. Bu nedenle bu nan değerleri 0'a dönüştürmek yerinde bir 
karar olacaktır.
"""

df["tenure"][df[df["TotalCharges"].isnull()].index]

df["TotalCharges"].fillna(0, inplace=True)


# Adım 7: Korelasyon analizi yapınız.

corr_matrix = df.corr()

"""
Veri setinde yer alan değişkenlerden TotalCharges ve tenure arasında iyi bir korelasyon vardır.
Bu iki değişken arasındaki korelasyon değeri 0.826'dır.
"""

df[["TotalCharges", "tenure"]].plot.scatter(x="TotalCharges", y="tenure")


##################################
# Görev 2: Feature Engineering
##################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

"""
Veri setinde aykırı değer bulunmamıştır. Eksik değerler ise Görev1 - Adım 6'da korelasyon hesabından önce 
doldurulmuştur.
"""

df.isnull().sum()

# Adım 2: Yeni değişkenler oluşturunuz.

df.loc[(df["PhoneService"] == "Yes") &
       (df["InternetService"] != "No") &
       (df["StreamingTV"] == "Yes") &
       (df["StreamingMovies"] == "Yes"), ["NEW_ALL_SERVICES"]] = "Yes"
df.NEW_ALL_SERVICES.fillna("No", inplace=True)

# Adım 3: Encoding işlemini gerçekleştiriniz.

df_new = pd.get_dummies(df,
                        columns=["gender", "Partner", "Dependents", "PhoneService",
                                 "MultipleLines", "InternetService", "OnlineSecurity",
                                 "OnlineBackup", "DeviceProtection", "TechSupport",
                                 "StreamingTV", "StreamingMovies", "Contract",
                                 "PaperlessBilling", "PaymentMethod", "NEW_ALL_SERVICES"],
                        drop_first=True)

le = LabelEncoder()
df_new["Churn"] = le.fit_transform(df_new["Churn"])


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()
df_new[num_cols] = rs.fit_transform(df_new[num_cols])


# Adım 5: Model oluşturunuz.

y = df_new["Churn"]
X = df_new.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


