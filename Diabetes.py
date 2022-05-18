##################################
# Feature Engineering
##################################

# İş Problemi:
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadığını tahmin edilecek bir makine öğrenmesi
# modeli geliştirilmesi istenmektedir. Modeli gelişitirmeden önce gerekli olan veri analizi ve özellik mühendisliği
# adımlarını gerçekleştirmeniz beklenmektedir.

# Veri Seti Hikayesi:
# Pregnancies: Hamilelik sayısı
# Glucose: Oral glikoz tolerans tetinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure: Kan basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness: Cilt kalınlığı
# Insulin: 2 saatlik serum insülini (mm U/ml)
# DiabetesPedigreeFunction: Fonsksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI: Vücut kitle endeksi
# Age: Yaş (yıl)
# Outcome: Hastalığa sahip (1) ya da değil (0)


##################################
# Görev 1: Keşifçi Veri Analizi
##################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Adım 1: Genel resmi inceliyiniz.

df = pd.read_csv("week6/Hw6/diabetes/diabetes.csv")
df.rename(columns={"DiabetesPedigreeFunction": "DPF"}, inplace=True)  # isim uzun olduğu için değiştirildi.

df.head()
df.dtypes
df.shape  # (768, 9)
df.isnull().sum()  # 0
df.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
sns.pairplot(df, hue="Outcome")
plt.savefig("week6/Hw6/diabetes/diabetes_pairplot.png")

"""
Glucose, BloodPressure, SkinThickness, Insulin, BMI değerleri içerisinde 0 değerleri var bunlar eksik veri mi?
Yoksa sıfır olma ihtimali olan veriler mi araştırılmalı.
"""

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

df.Outcome.value_counts()

"""
Tek kategorik değişken olan Outcome değişkeni kişinin hastalığa sahip olup olmadığını göstermektedir.
Eksik veya aykırı değer bulunmamaktadır 0 ve 1 değerlerinden oluşmaktadır.
"""

df[num_cols].describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T

"""
Numerik değişkenlerde daha önce de belirtildiği gibi eksik değerler bulunmaktadır. Glucose, BloodPressure, 
SkinThickness, Insulin, BMI gibi değerlerde sıfır değerinin bulunması çok olası olmadığından eksik değer 
olarak değerlendirilmelidir.
"""

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)

"""
Bir adet kategorik değişken bulunduğundan ve bu değişken aynı zaman hedef değişken olduğundan değerlendirmek 
anlamlı olmayacaktır.
"""


def num_cols_with_target(dataframe, num_cols, target_name):
    for num_col in num_cols:
        print(num_col.center(20, "_"))
        print(pd.DataFrame({"MEAN": dataframe.groupby(target_name)[num_col].mean(),
                            "COUNT": dataframe.groupby(target_name)[num_col].count()}))
        print("\n")


num_cols_with_target(df, num_cols, "Outcome")


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
Dikkate değer bir aykırılık bulunmamaktadır.
"""

# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()  # NaN değer bulunmamaktadır.

print("Number of zero variables")
for col_name in num_cols:
    print(f"{col_name}: {df[df[col_name] == 0].shape[0]}")

"""
Glucose, BloodPressure, BMI gibi değişkenlerin 0 olma ihtimali yoktur. Bu nedenle bu değişkenlerde eksik veri olduğu
söylenebilir.
"""

# Adım 7: Korelasyon analizi yapınız.

corr_matrix = df.corr()

"""
Veri setinde yer alan değişkenler arasında iyi bir korelasyon bulunamamıştır.
En iyi korelasyon değeri 0.544 ile Age - Pregnancies arasındadır.
"""


##################################
# Görev 2: Feature Engineering
##################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama
# Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin;
# bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumudikkate alarak sıfır değerlerini ilgili değerlerde
# NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

nan_cols = [col for col in num_cols if "Pregnancies" not in col]
"""
Numerik kolonlar içerisinde yer alan ve 0 olması ihtimali olan Pregnancies değeri nan_cols listesi içine alınmamıştır.
"""

df[df[nan_cols] == 0] = np.nan
df.isnull().sum()


df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace=True)
df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].mean(), inplace=True)



# Adım 2: Yeni değişkenler oluşturunuz.

# referance: https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm
df.loc[(df["BMI"] < 18.5), "NEW_BMI"] = "underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_BMI"] = "normal"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), "NEW_BMI"] = "overweight"
df.loc[(df["BMI"] >= 30) & (df["BMI"] < 35), "NEW_BMI"] = "obese"
df.loc[(df["BMI"] >= 35), "NEW_BMI"] = "extremly_obese"
df.NEW_BMI.value_counts()

# referance: https://emedicine.medscape.com/article/2089224-overview
df.loc[(df["Insulin"] > 25) & (df["Insulin"] < 166), "NEW_INSULIN"] = "normal"
df.loc[(df["Insulin"] <= 25) | (df["Insulin"] >= 166), "NEW_INSULIN"] = "abnormal"
df.NEW_INSULIN.value_counts()

# referance: https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451#:~:text=A%20blood%20sugar%20level%20less,mmol%2FL)%20indicates%20prediabetes.
df.loc[(df["Age"] > 45) & (df["BMI"] > 23), "NEW_MAYO_RISK"] = "risk-group"
df.NEW_MAYO_RISK.fillna("no-risk", inplace=True)
df.NEW_MAYO_RISK.value_counts()


# Adım 3: Encoding işlemini gerçekleştiriniz.

df_new = pd.get_dummies(df, columns=["NEW_BMI", "NEW_INSULIN", "NEW_MAYO_RISK"], drop_first=True)


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()
df_new[num_cols] = rs.fit_transform(df_new[num_cols])


# Adım 5: Model oluşturunuz.

y = df_new["Outcome"]
X = df_new.drop(["Outcome", "BMI", "Insulin"], axis=1)
# BMI ve Insulin için yeni değişkenler oluşturulduğu için eski değişkenleri çıkarıldı.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)





















