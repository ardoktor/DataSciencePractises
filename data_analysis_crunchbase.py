import pandas as pd
import numpy as np

df = pd.read_csv('investments_VC.csv', encoding= 'unicode_escape')
df.head()

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Unique #####################")
    print(dataframe.nunique())
check_df(df)

df.columns = df.columns.str.replace(' ', '')
df.columns

df['funding_total_usd'] = df['funding_total_usd'].str.replace(',','')
df['funding_total_usd'].head(15)

df['funding_total_usd'] = df['funding_total_usd'].str.replace('-','0')
df['funding_total_usd'].head(15)

df['funding_total_usd']=df['funding_total_usd'].str.replace(' ','')
df['funding_total_usd'].head(15)

df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'])

df = df.drop(["founded_month","founded_quarter","founded_year"],axis=1)

df['first_funding_at'] = pd.to_datetime(df["first_funding_at"], format="%Y-%m-%d", exact=False, errors='coerce')
df['last_funding_at'] = pd.to_datetime(df["last_funding_at"], format="%Y-%m-%d", exact=False, errors='coerce')
df['founded_at'] = pd.to_datetime(df["founded_at"], format="%Y-%m-%d", exact=False, errors='coerce')
df.dtypes


country = pd.read_csv('wikipedia-iso-country-codes.csv')
country.head()

country = pd.read_csv('/kaggle/input/iso-country-codes-global/wikipedia-iso-country-codes.csv')
country.head()

country = country[['English short name lower case','Alpha-3 code']]
country.head()

country.columns = ['country','country_code']
country.head()

df = df.merge(country,on='country_code',how='left')
df.shape

df = df.drop_duplicates()
df.shape




df.groupby('status')['name'].nunique()
df.groupby('status')['funding_rounds'].describe()

df.groupby('status')[['funding_rounds', 'funding_total_usd', 'seed', 'venture', 'equity_crowdfunding',
       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt',
       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']].mean().T

df.groupby('country')['name'].nunique().sort_values(ascending=False).head(30)
df.groupby('region')['name'].count().sort_values(ascending = False).head(10)

df = df.drop(["state_code"],axis=1)
df.dtypes

df[['region','country_code']].isnull().sum()

df = df.dropna(subset = ['region'])
df[(df['country_code'].isnull() & df['country'].isnull() & df['city'].isnull() & df['region'].isnull())]

df[df['country']=='Turkey']['permalink'].count()

df = df.dropna(subset = ['status'])
df.shape

df1 = df.copy()
df1.shape
df1['founded_at'].hist()

df[df["founded_at"]==df['founded_at'].max()]
df[df["founded_at"]==df['founded_at'].min()]
df.groupby('market')['funding_total_usd'].sum().sort_values(ascending = False).head(5)


df.groupby('market')['name'].count().sort_values(ascending = False).head(5)
df.groupby('region')['name'].count().sort_values(ascending = False).head(10)

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols

non_cat_list = ['permalink','name','homepage_url']
for each in non_cat_list:
    cat_cols.remove(each)
cat_cols

numerical_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in cat_cols]

numerical_cols

import matplotlib.pyplot as plt


import seaborn as sns
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col,plot = False)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[[numerical_col]].describe(quantiles).T, end="\n\n")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True) #block=True olduğunda grafikleri üst üste değil, arka arkaya çıkarır

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

for col in numerical_cols:
    target_summary_with_num(df, "status", col)

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.loc[:, df.columns != 'permalink'].corr(numeric_only=True), annot=True, fmt=".2f", ax=ax, cmap="YlOrBr")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


df['diff_funding'] = df['last_funding_at'] - df['first_funding_at']
df['diff_funding'].describe()


df['diff_funding_months'] = (df['last_funding_at'] - df['first_funding_at'])/np.timedelta64(1, 'M')
df['diff_funding_months'].describe()


df['diff_first_funding_months'] = (df['first_funding_at'] - df['founded_at'])/np.timedelta64(1, 'M')
df['diff_first_funding_months'].describe()


print(df['diff_first_funding_months'].describe())
plt.figure(figsize=(9,4))
sns.distplot(df['diff_first_funding_months'], color='g', bins=150, hist_kws={'alpha': 0.4});

df1.groupby(df['diff_funding'])['permalink'].count().sort_index(ascending = True).head(50)
df['market'].nunique()
df['market'].value_counts()
df['country'].nunique()

df.groupby('status')['angel'].mean().sort_values(ascending = False).head(15)
df.groupby('country')['grant'].mean().sort_values(ascending = False).head(15)

df.groupby('status')['convertible_note'].mean().sort_values(ascending = False).head(15)


round_cols = [col for col in df.columns if 'round' in col]
round_cols.remove('funding_rounds')
round_cols

rounds_df = df[round_cols].apply(lambda x: (x > 0).astype(int))
rounds_df['total'] = rounds_df['round_A'] +rounds_df['round_B'] +rounds_df['round_C'] +rounds_df['round_D'] +rounds_df['round_E'] +rounds_df['round_F'] +rounds_df['round_G'] +rounds_df['round_H']
rounds_df

df['round_A_H_total'] = rounds_df['total']
df.head()


df['angel_status'] = (df['angel'] > 0).astype(int)
df.head()

df['grant_status'] = (df['grant'] > 0).astype(int)
df.head()

df['avg_fund_size'] = df['funding_total_usd'] / df['funding_rounds']
df['avg_fund_size'].head()

df['ratio_seed_tot'] =df['seed'] / df['funding_total_usd']
df['ratio_seed_tot']


df['ratio_seed_tot'].isnull().sum()
df['ratio_debt_tot'] =df['debt_financing'] / df['funding_total_usd']

df['ratio_debt_tot'].isnull().sum()

df['convertible_status'] = (df['convertible_note'] > 0).astype(int)
df.head()

df["seed_quartiles"] = pd.qcut(df['seed'],4, duplicates='drop')
df["seed_quartiles"]

seed_labels=["Low_seed","High_seed"]
df["seed_quartiles"] = pd.qcut(df['seed'],4,labels = seed_labels,duplicates='drop')
df["seed_quartiles"].value_counts()

df["angel"].value_counts().sort_index(ascending = False)

df.loc[(df["angel"]<= 40000000)& (df["angel"]>= 1000000),"angel_degree"] = "High_angel"
df.loc[(df["angel"]< 1000000)& (df["angel"]>= 500000),"angel_degree"] = "Middle_angel"
df.loc[(df["angel"]< 500000)& (df["angel"]>= 100000),"angel_degree"] = "MiddleLow_angel"
df.loc[(df["angel"]< 100000)& (df["angel"]>= 10000),"angel_degree"] = "Low_angel"


df[df["angel_degree"] == 'Low_angel'].count()


df.loc[(df["funding_total_usd"]<= 30079503010.0)& (df["funding_total_usd"]>= 100000000),"tot_funding_degree"] = "High_fund"
df.loc[(df["funding_total_usd"]< 100000000)& (df["funding_total_usd"]>= 10000000),"tot_funding_degree"] = "Middle_fund"
df.loc[(df["funding_total_usd"]< 10000000)& (df["funding_total_usd"]>= 1000000),"tot_funding_degree"] = "MiddleLow_fund"
df.loc[(df["funding_total_usd"]< 1000000),"tot_funding_degree"] = "low_fund"
df['tot_funding_degree'].value_counts()


df['venture'].describe().max()

df.loc[(df["venture"]<= 2351000001.0)& (df["venture"]>= 500000000),"venture_degree"] = "High_venture"
df.loc[(df["venture"]< 500000000)& (df["venture"]>= 10000000),"venture_degree"] = "Middle_venture"
df.loc[(df["venture"]< 10000000)& (df["venture"]>= 1000000),"venture_degree"] = "MiddleLow_venture"
df.loc[(df["venture"]< 1000000),"venture_degree"] = "Low_venture"
df['venture_degree'].value_counts()

df.loc[(df["seed_quartiles"]== 'High_seed')& (df["angel_degree"]== 'High_angel'),"start_postion"] = "perfect_start"
df.loc[(df["seed_quartiles"]== 'High_seed')& (df["angel_degree"]== 'Middle_angel'),"start_postion"] = "good_start"
df.loc[(df["seed_quartiles"]== 'Low_seed')& (df["angel_degree"]== 'MiddleLow_angel'),"start_postion"] = "avarage_start"
df.loc[(df["seed_quartiles"]== 'Low_seed')& (df["angel_degree"]== 'Low_angel'),"start_postion"] = "poor_start"

df['secondary_status'] = (df['secondary_market'] > 0).astype(int)
df.head()
df_after_features_created = df

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


df.loc[(df['start_postion'].isnull()),'start_postion'] = 'not_significant'
df.loc[(df['angel_degree'].isnull()),'angel_degree'] = 'not_angel_inv' # angel yoksa doğarl olarak oraya null yazdı
med_date = df['founded_at'].median()

df['founded_at'].fillna(med_date, inplace=True)
df['diff_first_funding_months'] = (df['first_funding_at'] - df['founded_at'])/np.timedelta64(1, 'M')
# burayı baştan yapıyoruz tabi

na_columns = missing_values_table(df, na_name=True)

df = df.dropna(subset=['category_list'])
na_columns = missing_values_table(df, na_name=True)

na_columns = missing_values_table(df, na_name=True)

df = df.dropna(subset=['first_funding_at'])
na_columns = missing_values_table(df, na_name=True)

df = df.dropna(subset=['country'])
na_columns = missing_values_table(df, na_name=True)


df = df.dropna(subset=['market'])
na_columns = missing_values_table(df, na_name=True)

df['ratio_seed_tot'].fillna(0, inplace=True)

df['ratio_debt_tot'].fillna(0, inplace=True)
na_columns = missing_values_table(df, na_name=True)

import datetime as dt
max_last_fund_date = df['last_funding_at'].max()

df['recency'] = (max_last_fund_date- df["founded_at"])/np.timedelta64(1, 'M')
df.head()

df.to_pickle('report_df.pkl')
non_usable_features = ["permalink","name","homepage_url","category_list","country_code", # burada country kullanacağım direct
                       "region","city","founded_at","first_funding_at","last_funding_at","diff_funding"] # diff funding'e de gerek yok çünkü zaten months oalrak kullanıyorum


ml_features = [col for col in df.columns if col not in non_usable_features]
ml_features

ml_df = df[ml_features]
ml_df.to_pickle('pickle_clean_crunchbase.pkl')

ml_df.loc[df["status"]== 'acquired',"target"] = 1
ml_df.loc[df["status"]== 'closed',"target"] = 0
ml_df.loc[df["status"]== 'operating',"target"] = 2
ml_df.head()

# Now I can drop status
ml_df = ml_df.drop('status',axis=1)
ml_df.head()


from sklearn.preprocessing import LabelEncoder
ordinal_ranking = ['not_angel_inv','Low_angel','MiddleLow_angel','Middle_angel','High_angel']
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_ranking)
ml_df['encoded_angel_degree'] = label_encoder.transform(df['angel_degree'])

ml_df = ml_df.drop('angel_degree',axis=1)
ml_df.head()

ordinal_ranking = ['low_fund','MiddleLow_fund','Middle_fund','High_fund']
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_ranking)
ml_df['encoded_tot_funding_degree'] = label_encoder.transform(df['tot_funding_degree'])
ml_df = ml_df.drop('tot_funding_degree',axis=1)
ml_df.head()

ordinal_ranking = ['Low_venture','MiddleLow_venture','Middle_venture','High_venture']
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_ranking)
ml_df['encoded_venture_degree'] = label_encoder.transform(df['venture_degree'])
ml_df = ml_df.drop('venture_degree',axis=1)
ml_df.head()

ordinal_ranking = ['not_significant','poor_start','avarage_start','good_start','perfect_start']
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_ranking)
ml_df['start_degree'] = label_encoder.transform(df['start_postion'])
ml_df = ml_df.drop('start_postion',axis=1)
ml_df.head()

ml_df['market'].value_counts().sort_values(ascending = False).head(66).sum()/ len(ml_df)
main_categories = list(ml_df['market'].value_counts().sort_values(ascending = False).head(66).index)

# Remove leading and trailing white spaces from each category
cleaned_categories = [category.strip() for category in main_categories]
ml_df['market'] = ml_df['market'].str.strip()
len(cleaned_categories)

ml_df.loc[~ml_df['market'].isin(cleaned_categories), 'market'] = 'others'
ml_df['market'].value_counts()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ml_df = one_hot_encoder(ml_df,['market','country'], drop_first= False)

'country' in ml_df.columns

ml_df['seed_quartiles'].value_counts()

ordinal_ranking = ['Low_seed','High_seed']
label_encoder = LabelEncoder()
label_encoder.fit(ordinal_ranking)
ml_df['seed_degree'] = label_encoder.transform(df['seed_quartiles'])
ml_df = ml_df.drop('seed_quartiles',axis=1)
ml_df.head()

ml_df.isnull().sum().any()
ml_df.to_pickle('model_ready_ds.pkl')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

numeric_cols = ml_df.select_dtypes(include=['int', 'float']).columns.tolist()

print(numeric_cols)

col_dont_want = ['angel_status', 'grant_status', 'convertible_status', 'secondary_status', 'target']

for each in col_dont_want:
    numeric_cols.remove(each)

'grant_status' in numeric_cols

ml_df[numeric_cols] = scaler.fit_transform(ml_df[numeric_cols])
ml_df[numeric_cols].head()

(ml_df['target'].value_counts()/ ml_df.shape[0])*100

# Separate the majority and minority classes
minority_classes = ml_df[ml_df['target'].isin([0, 1])]  # Assuming classes 0 and 1 are minority classes
majority_class = ml_df[ml_df['target'] == 2]  # Assuming class 2 is the majority class

# Randomly select instances from the majority class
num_minority_instances = len(minority_classes)
undersampled_majority = majority_class.sample(n=num_minority_instances, random_state=42)

# Create a balanced dataset by combining the minority and undersampled majority classes
balanced_df = pd.concat([minority_classes, undersampled_majority])

# Now, balanced_df contains your downsampled, balanced dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
y = balanced_df['target']
X = balanced_df.drop("target", axis=1)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)


rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 2)}")

# Recall (also known as sensitivity or true positive rate)
recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"Recall: {round(recall, 3)}")

# Precision
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"Precision: {round(precision, 2)}")

# F1 Score (the harmonic mean of precision and recall)
f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"F1: {round(f1, 2)}")

# AUC (Area Under the Receiver Operating Characteristic Curve) is not directly applicable to multiclass problems.
# If you want to compute AUC, you might need to use one-vs-all (OvA) or one-vs-one (OvO) strategies.
# Here, I'll show you how to compute it using a one-vs-all approach.
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Convert multiclass labels to binary format for ROC AUC calculation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels

# Compute ROC AUC for each class separately and then average them
roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
print(f"AUC: {round(roc_auc, 2)}")


from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  cross_val_score

models = [('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVC', SVC()),
          ('GBM', GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier(objective='multi:softmax'))]

for name, classifier in models:
    print(f"MODEL:({name})")
    print("   ")
    ml_model = classifier.fit(X_train, y_train)
    y_pred = ml_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(accuracy, 2)}")

    # Recall (also known as sensitivity or true positive rate)
    recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    print(f"Recall: {round(recall, 3)}")

    # Precision
    precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    print(f"Precision: {round(precision, 2)}")

    # F1 Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    print(f"F1: {round(f1, 2)}")

    # AUC (Area Under the Receiver Operating Characteristic Curve) is not directly applicable to multiclass problems.
    # If you want to compute AUC, you might need to use one-vs-all (OvA) or one-vs-one (OvO) strategies.
    # Here, I'll show you how to compute it using a one-vs-all approach.
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    # Convert multiclass labels to binary format for ROC AUC calculation
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels
    y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels

    # Compute ROC AUC for each class separately and then average them
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
    print(f"AUC: {round(roc_auc, 2)}")
    print("   ")

base_model = GradientBoostingClassifier(random_state=46).fit(X_train, y_train)
y_pred = base_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 2)}")

# Recall (also known as sensitivity or true positive rate)
recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"Recall: {round(recall, 3)}")

# Precision
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"Precision: {round(precision, 2)}")

# F1 Score (the harmonic mean of precision and recall)
f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
print(f"F1: {round(f1, 2)}")

# AUC (Area Under the Receiver Operating Characteristic Curve) is not directly applicable to multiclass problems.
# If you want to compute AUC, you might need to use one-vs-all (OvA) or one-vs-one (OvO) strategies.
# Here, I'll show you how to compute it using a one-vs-all approach.
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Convert multiclass labels to binary format for ROC AUC calculation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])  # Replace [0, 1, 2] with your actual class labels

# Compute ROC AUC for each class separately and then average them
roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
print(f"AUC: {round(roc_auc, 2)}")


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import warnings
warnings.simplefilter(action="ignore")

# Define a custom scoring function for ROC AUC that accounts for multiclass problems
"""
def custom_roc_auc(y_true, y_pred):
    y_true_bin = label_binarize(y_true, classes=list(range(len(np.unique(y_true)))))
    y_pred_bin = label_binarize(y_pred, classes=list(range(len(np.unique(y_true)))))

    # Debugging prints
    print("y_true_bin:", y_true_bin)
    print("y_pred_bin:", y_pred_bin)

    return roc_auc_score(y_true_bin, y_pred_bin, average='macro')
"""
# Define the list of scoring metrics, including the custom ROC AUC function
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Perform cross-validation
cv_results = cross_validate(
    base_model,
    X,
    y,
    cv=5,
    scoring=scoring_metrics,
    return_train_score=False
)

# Access the results for each scoring metric
accuracy_scores = cv_results['test_accuracy']
precision_scores = cv_results['test_precision']
recall_scores = cv_results['test_recall']
f1_scores = cv_results['test_f1']

# Print the results
print(f"Accuracy Scores: {accuracy_scores}")
print(f"Precision Scores: {precision_scores}")
print(f"Recall Scores: {recall_scores}")
print(f"F1 Scores: {f1_scores}")

"""
param_dist = {
    'n_estimators': np.arange(100, 1000, 50),  # Number of boosting stages
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],  # Learning rate
    'max_depth': [3, 4, 5, 6, 7],  # Maximum depth of the individual trees
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Fraction of samples used for fitting the trees
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings that are sampled
    scoring='neg_log_loss',  # Use the log loss as the evaluation metric
    cv=5,  # Number of cross-validation folds
    verbose=2,  # Controls the verbosity: higher values show more information
    n_jobs=-1  # Use all available CPU cores for parallel computation
)

# Perform the random search
random_search.fit(X, y)

# Print the best hyperparameters and their corresponding score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score (Neg Log Loss):", -random_search.best_score_)

"""





"""
# I set around {'subsample': 0.8, 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1}
param_dist = {
    'n_estimators': np.arange(100, 200, 50),  # Number of boosting stages 
    'learning_rate': [0.05, 0.1],  # Learning rate 2
    'max_depth': [4, 5],  # Maximum depth of the individual trees 2
    'subsample': [0.7, 0.8],  # Fraction of samples used for fitting the trees 2
}

# Create the RandomizedSearchCV object
grid_search = GridSearchCV(
    base_model,
    param_grid=param_dist,
    scoring='neg_log_loss',  # Use the log loss as the evaluation metric
    cv=5,  # Number of cross-validation folds
    verbose=2,  # Controls the verbosity: higher values show more information
    n_jobs=-1  # Use all available CPU cores for parallel computation
)

# Perform the random search
grid_search.fit(X, y)

# Print the best hyperparameters and their corresponding score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score (Neg Log Loss):", -grid_search.best_score_)


grid_search_best_params= grid_search.best_params_
grid_search_best_params
"""


grid_search_best_params = {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 150, 'subsample': 0.7}
final_model = base_model.set_params(**grid_search_best_params).fit(X, y)

# Define the list of scoring metrics, including the custom ROC AUC function
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
}

# Perform cross-validation
cv_results = cross_validate(
    final_model,
    X,  # Your feature matrix
    y,  # Your target vector
    cv=5,  # Number of cross-validation folds
    scoring=scoring_metrics,
    return_train_score=False
)

# Access the results for each scoring metric
accuracy_scores = cv_results['test_accuracy']
precision_scores = cv_results['test_precision']
recall_scores = cv_results['test_recall']
f1_scores = cv_results['test_f1']  # Access neg_log_loss scores

# Print the results
print(f"Accuracy Scores: {accuracy_scores.mean()}")
print(f"Precision Scores: {precision_scores.mean()}")
print(f"Recall Scores: {recall_scores.mean()}")
print(f"F1 Scores: {f1_scores.mean()}")
print(f"Negative Log Loss Scores: {-neg_log_loss_scores.mean()}")

final_model = base_model.set_params(**grid_search_best_params).fit(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have already made predictions with 'y_test_pred' and you have ground truth labels 'y_test'

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision (weighted)
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall (weighted)
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1 score (weighted)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print or use the metrics as needed
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X)

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})
most_important= feature_imp.sort_values("Value",ascending=False)[:20]

plt.figure(figsize=(10, 10))
sns.set(font_scale=1)
sns.barplot(x="Value", y="Feature", data=most_important)
plt.title('Features')
plt.tight_layout()
plt.show()

feature_imp.to_csv('feature_imp.csv', index=False)









