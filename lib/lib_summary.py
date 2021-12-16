from sklearn import preprocessing, impute, utils, model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler

## 1.5. Moves columns into a df => front to end
'''
Moves columns into a df.
:parameter
    :param dtf: dataframe - input data
    :param lst_cols: list - names of the columns that must be moved
    :param where: str - "front" or "end"
:return
    df with moved columns
'''
def pop_columns(df, lst_cols, where="front"):
    current_cols = df.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        df = df[lst_cols + current_cols]
    elif where == "end":
        df = df[current_cols + lst_cols]
    return df
## 3.7. Computes all the required data preprocessing.
'''
Computes all the required data preprocessing.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable 
    :param processNas: str or None - "mean", "median", "most_frequent"
    :param processCategorical: str or None - "dummies"
    :param split: num or None - test_size (example 0.2)
    :param scale: str or None - "standard", "minmax"
    :param task: str - "classification" or "regression"
:return
    dictionary with dtf, X_names lsit, (X_train, X_test), (Y_train, Y_test), scaler
'''
def data_preprocessing(dtf, y, processNas=None, processCategorical=None, split=None, scale=None, task="classification"):
    try:
        dtf = pop_columns(dtf, [y], "front")
        
        ## missing
        ### check
        print("--- check missing ---")
        if dtf.isna().sum().sum() != 0:
            cols_with_missings = []
            for col in dtf.columns.to_list():
                if dtf[col].isna().sum() != 0:
                    print("WARNING:", col, "-->", dtf[col].isna().sum(), "Nas")
                    cols_with_missings.append(col)
            ### treat
            if processNas is not None:
                print("...treating Nas...")
                cols_with_missings_numeric = []
                for col in cols_with_missings:
                    if dtf[col].dtype == "O":
                        print(col, "categorical --> replacing Nas with label 'missing'")
                        dtf[col] = dtf[col].fillna('missing')
                    else:
                        cols_with_missings_numeric.append(col)
                if len(cols_with_missings_numeric) != 0:
                    print("replacing Nas in the numerical variables:", cols_with_missings_numeric)
                imputer = impute.SimpleImputer(strategy=processNas)
                imputer = imputer.fit(dtf[cols_with_missings_numeric])
                dtf[cols_with_missings_numeric] = imputer.transform(dtf[cols_with_missings_numeric])
        else:
            print("   OK: No missing")
                
        ## categorical data
        ### check
        print("--- check categorical data ---")
        cols_with_categorical = []
        for col in dtf.columns.to_list():
            if dtf[col].dtype == "O":
                print("WARNING:", col, "-->", dtf[col].nunique(), "categories")
                cols_with_categorical.append(col)
        ### treat
        if len(cols_with_categorical) != 0:
            if processCategorical is not None:
                print("...trating categorical...")
                for col in cols_with_categorical:
                    print(col)
                    dtf = pd.concat([dtf, pd.get_dummies(dtf[col], prefix=col)], axis=1).drop([col], axis=1)
        else:
            print("   OK: No categorical")
        
        ## 3.split train/test
        print("--- split train/test ---")
        X = dtf.drop(y, axis=1).values
        Y = dtf[y].values
        if split is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
            print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
            print("y_train mean:", round(np.mean(y_train),2), " | y_test mean:", round(np.mean(y_test),2))
            print(X_train.shape[1], "features:", dtf.drop(y, axis=1).columns.to_list())
        else:
            print("   OK: step skipped")
            X_train, y_train, X_test, y_test = X, Y, None, None
        
        ## 4.scaling
        print("--- scaling ---")
        if scale is not None:
            scalerX = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
            X_train = scalerX.fit_transform(X_train)
            scalerY = 0
            if X_test is not None:
                X_test = scalerX.transform(X_test)
            if task == "regression":
                scalerY = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
                y_train = scalerY.fit_transform(y_train.reshape(-1,1))
            print("   OK: scaled all features")
        else:
            print("   OK: step skipped")
            scalerX, scalerY = 0, 0
        
        return {"dtf":dtf, "X_names":dtf.drop(y, axis=1).columns.to_list(), 
                "X":(X_train, X_test), "y":(y_train, y_test), "scaler":(scalerX, scalerY)}
    
    except Exception as e:
        print("--- got error ---")
        print(e)


## 3.1. Chia dữ liệu train/ test
'''
Split the dataframe into train / test
shuffle: bool, default=True/ Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
return 2 tập dữ liệu dtf_train, dtf_test
'''
def dtf_partitioning(dtf, y, test_size=0.3, shuffle=True):
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    # Xem mức độ cân bằng dữ liệu
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    # Các giá trị trung bình của dữ liệu y với thuộc tính là regression
    print("y_train mean:", round(np.mean(dtf_train[y]),2), "| y_test mean:", round(np.mean(dtf_test[y]),2))
    # Các thuộc tính trong input
    print(dtf_train.shape[1], "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
'''
def evaluate_regr_model(y_test, predicted, figsize=(25,5)):
    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    
    ## residuals
    residuals = y_test - predicted
    #print(type(residuals))
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    #print('max_error: ',max_error)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    #print('max_idx: ', max_idx)
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    
    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
    
    ## Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()
    
    ## Plot residuals distribution
    sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax[2], label="mean = "+"{:,.0f}".format(np.mean(residuals)))
    ax[2].grid(True)
    ax[2].set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()



'''
Use shap to build an a explainer.
:parameter
    :param model: model instance (after fitting)
    :param X_names: list
    :param X_instance: array of size n x 1 (n,)
    :param X_train: array - if None the model is simple machine learning, if not None then it's a deep learning model
    :param task: string - "classification", "regression"
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_shap(model, X_names, X_instance, X_train=None, task="classification", top=10):
    ## create explainer
    ### machine learning
    if X_train is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
    ### deep learning
    else:
        explainer = shap.DeepExplainer(model, data=X_train[:100])
        shap_values = explainer.shap_values(X_instance.reshape(1,-1))[0].reshape(-1)

    ## plot
    ### classification
    if task == "classification":
        shap.decision_plot(explainer.expected_value, shap_values, link='logit', feature_order='importance',
                           features=X_instance, feature_names=X_names, feature_display_range=slice(-1,-top-1,-1))
    ### regression
    else:
        shap.waterfall_plot(explainer.expected_value[0], shap_values, 
                            features=X_instance, feature_names=X_names, max_display=top)


## A. LOAD DATA
### 1.1. Load data: dùng để load data các file đuôi csv, xlsx, json
'''
    file_dir: đường dẫn file năm ở đâu
    names: tên thuộc tính mong muốn
    return: kết quả là 1 dataFrame
'''
def loadData(file_dir="", names=""):
    #try:
        file_dir = file_dir.lower()
        if file_dir.endswith("csv"):
            df = pd.read_csv(file_dir, names=names)
            return df
        elif file_dir.endswith("xlsx"):
            df = pd.read_excel(file_dir, names=names)
            return df
        elif file_dir.endswith("json"):
            df = pd.read_json(file_dir, names=names)
            return df
        else:
            print("Please see file and path")


# 1.14. Chuyển dổi thuộc tính
'''
    Parameter:
        df: chọn dataframe
        lst_category: thuộc tính cần chuyển đổi
        names: tên côt
    Return
        Trả về 1 dataframe
'''
def change_feature_seriesToDataframe(df, lst_feature, names):
    df_length_ = df[lst_feature].to_frame(name=names)
    return df_length_

# 1.21. Xử lý date trong dataframe
'''
    parameter:
        - df: dataframe
        - feature_date: thuộc tính ngày
    return:
        - dataframe
'''
def date_add_feature(df, feature_date):
    df['fea_year'] = pd.DatetimeIndex(df[feature_date]).year
    df['fea_month'] = pd.DatetimeIndex(df[feature_date]).month
    df['fea_day'] = pd.DatetimeIndex(df[feature_date]).day
    df['fea_weekofyear'] = pd.DatetimeIndex(df[feature_date]).weekofyear
    df['fea_daily'] = pd.DatetimeIndex(df[feature_date]).weekday
    return df

# 1.14. Chuyển dổi thuộc tính
'''
    Parameter:
        df: chọn dataframe
        lst_category: thuộc tính cần chuyển đổi
        names: tên côt
    Return
        Trả về 1 dataframe
'''
def change_feature_seriesToDataframe(df, lst_feature, names):
    df_length_ = df[lst_feature].to_frame(name=names)
    return df_length_

# 1.22. Xử lý combine data feature
'''
    parameter: 
        - lst_concat: là 1 list các dataframe
    return:
        - Trả về 1 dataframe mới
'''
def dataframe_concat(lst_concat = []):
    df_new = pd.concat(lst_concat, axis=1)
    return df_new

### 1.14.1. Chuyển đổi kiểu dữ liệu cho thuộc tính: change astype for feature
'''
    prameter: 
        - df: là dữ liệu dataframe
        - lst_int: Là dữ liệu muốn chuyển qua kiểu dữ liệu int
        - lst_float: là dữ liệu muốn chuyển qua kiễu dữ liệu float
    return trả về dataframe mong muốn
'''
def change_type_lst(df, lst_change, choose = 'int'):
    if choose == 'int':
        for i in lst_change:
            df[i] = df[i].astype(int)
    elif choose == 'float':
        for i in lst_change:
            df[i] = df[i].astype(float)

### 1.14.1. Chuyển đổi kiểu dữ liệu cho thuộc tính: change astype for feature
'''
    prameter: 
        - df: là dữ liệu dataframe
        - lst_float: thuộc tính muốn chuyển qua 
    return trả về dataframe mong muốn
'''
def changeToAstype_date(df, feature_date):
    # Chuyển thành object sang qua date
    df[feature_date] = pd.to_datetime(df[feature_date])
    return df

## 3.4. Transforms a categorical column into dummy columns
### Dùng để chuyển về số bằng dummy trong Feature engineering
'''
Transforms a categorical column into dummy columns
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dropx: logic - whether the x column should be dropped
:return
    dtf with dummy columns added
'''
def add_dummies(dtf, x, dropx=False):
    dtf_dummy = pd.get_dummies(dtf[x], prefix=x, drop_first=True, dummy_na=False)
    dtf = pd.concat([dtf, dtf_dummy], axis=1)
    print( dtf.filter(like=x, axis=1).head() )
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf


### 3.2. Robust scaler: Phân phổi không chuẩn hoặc không xấp xỉ chuẩn; và có outlier
### 3.2.1. Robust scaler
##### Trả về dữ liệu là 1 dataframe sau đó dùng concat để đưa vào df
def robust_Scaler(df, lst_lientuc_chosen):
    try:
        # Thêm tên khác cho các thuộc tinh scaler
        lst_name_column = []
        for i in lst_lientuc_chosen:
            lst_name_column.append(i+'_scaler')
        # Chuẩn hoá bằng RobustScaler trên dữ liệu đã Log normalization
        #--> Do các thuộc tính không có PP chuẩn và có outliers nên không sử dụng StandarScaler/MinMaxScaler
        scaler = RobustScaler()
        data = df[lst_lientuc_chosen]
        # X_train_scale = scaler.fit_transform(X_before_scale)
        scaler = scaler.fit(data)
        df_new = scaler.transform(data)
        df_new = pd.DataFrame(df_new, columns=lst_name_column)
        return df_new
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")