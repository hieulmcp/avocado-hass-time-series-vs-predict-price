STATISTICS_AND_LINEAR_ALGEBRA_PRELIMINARIES = '''
    6.3. Measurement Formula (Page: 59)
        - Mean absolute error (MAE): Có nghĩa là thước đo giữa 2 biến liên tục (biến dự đoán và biến liên tục)
        - Mean squared error (MSE): Bình phương của Mean absolute error
        - Root Mean squared error (RMSE): căn bậc 2 MSE
        - Total sum of squares (TTS): tổ hợp các giá trị y thực tế vs y trung bình bình phương
        - Explained Sum of Squares (ESS): Tổ hợp các giá trị dự đoán vs y trung bình bình phương
        - Residual sum of squares (RSS): tổ hợp y dự đoán vs y trung bình bình phương
        - R2 - Coefficient of determination: r2 = 1 - RSS/TSS= ESS/TSS
    6.4. Confusion matrix (page number: 60)
        - Recall
        - Precision
        - Accuracy
        - 𝐹1-score
    6.5 Statistical Tests (page number: 61)
        6.5.1 Correlational Test
            • Pearson correlation: Tests for the strength of the association between two continuous variables.
            • Spearman correlation: Tests for the strength of the association between two ordinal variables (does
            not rely on the assumption of normal distributed data).
            • Chi-square: Tests for the strength of the association between two categorical variables.
        6.5.2 Comparison of Means test
            • Paired T-test: Tests for difference between two related variables.
            • Independent T-test: Tests for difference between two independent variables.
            • ANOVA: Tests the difference between group means after any other variance in the outcome variable
            is accounted for.
        6.5.3 Non-parametric Test
            • Wilcoxon rank-sum test: Tests for difference between two independent variables - takes into account
            magnitude and direction of difference.
            • Wilcoxon sign-rank test: Tests for difference between two related variables - takes into account magnitude
            and direction of difference.
            • Sign test: Tests if two related variables are different – ignores magnitude of change, only takes into
            account direction
        
'''

DATA_EXPLORATION = ''' 
    7.1. Univariate Analysis (page: 63)
        7.1.1 Numerical Variables
            - Describe
                    + The describe function in pandas and spark will give us most of the statistical results, such as min, median, max, quartiles and standard deviation. 
                    With the help of the user defined function, you can get even more statistical results.
            - Skew, Kru
        7.1.2 Categorical Variables
            - Compared with the numerical variables, the categorical variables are much more easier to do the exploration
'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import warnings
import pandas_profiling as pp # tổng quan ban đầu về dữ liệu => Cài trên này
from matplotlib import pyplot as plt

from pyspark.sql.functions import col, skewness, kurtosis


from pyspark.sql import functions as F
from pyspark.sql.functions import rank,sum,col
from pyspark.sql import Window



#A. Univariate Analysis
## 1.1. Thông kê biến liên tục trong numerical variable
'''
    Function to union the basic stats results and deciles
    :param df_in: the input dataframe
    :param columns: the cloumn name list of the numerical variable
    :param deciles: the deciles output
    :return : the numerical describe info. of the input dataframe
'''

def describe_pd(df_in, columns, deciles=False):

    if deciles:
        percentiles = np.array(range(0, 110, 10))
    else:
        percentiles = [25, 50, 75]
        
    percs = np.transpose([np.percentile(df_in.select(x).collect(),
    percentiles) for x in columns])
    percs = pd.DataFrame(percs, columns=columns)
    percs['summary'] = [str(p) + '%' for p in percentiles]
    spark_describe = df_in.describe().toPandas()
    new_df = pd.concat([spark_describe, percs],ignore_index=True)
    new_df = new_df.round(2)
    
    return new_df[['summary'] + columns]

### 1.2. Thông kê hàm lệch phải lệch trái và  nhón hay bẹt
'''
    df: dataframe
    var: biến
'''
def skew_kur_var(df, var):
    return df.select(skewness(var),kurtosis(var)).show()

### 1.3. Histograms valization
def histograms_valization(df, var):
    x = df[var]
    bins = np.arange(0, 100, 5.0)
    plt.figure(figsize=(10,8))
    # the histogram of the data
    plt.hist(x, bins, alpha=0.8, histtype='bar', color='gold', ec='black',weights=np.zeros_like(x) + 100. / x.size)
    plt.xlabel(var)
    plt.ylabel('percentage')
    plt.xticks(bins)
    plt.show()
    plt.savefig(var+".pdf", bbox_inches='tight')

### 1.4. Frequency table- bảng tần số biên categorical variable
def frequency_table_categorical_variable(df, var_groupby, var_statis):
    window = Window.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
    tab = df.select([var_groupby,var_statis]).\
        groupBy(var_groupby).\
        agg(F.count(var_statis).alias('num'),
            F.mean(var_statis).alias('avg'),
            F.min(var_statis).alias('min'),
            F.max(var_statis).alias('max')).\
            withColumn('total',sum(col('num')).over(window)).\
            withColumn('Percent',col('Credit_num')*100/col('total')).\
            drop(col('total'))
    return tab

#B. Multivariate Analysis
##B1. Numerical V.S. Numerical
## 2.1. Correlation matrix

from pyspark.mllib.stat import Statistics
import pandas as pd
def correlation_maxtrix(df, num_cols):
    corr_data = df.select(num_cols)
    col_names = corr_data.columns
    features = corr_data.rdd.map(lambda row: row[0:])
    corr_mat=Statistics.corr(features, method="pearson")
    corr_df = pd.DataFrame(corr_mat)
    corr_df.index, corr_df.columns = col_names, col_names
    return corr_df.to_string()

##B2. Categorical V.S. Categorical
## 2.2. Pearson’s Chi-squared test
### Warning: pyspark.ml.stat is only available in Spark 2.4.0.
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
def chi_squared_categorical(df, features, label):
    
    r = ChiSquareTest.test(df, features, label).head()
    print("pValues: " + str(r.pValues))
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    print("statistics: " + str(r.statistics))


#B3. Numerical V.S. Categorical



DATA_MANIPULATION_FEATURES = '''
    8.1 Feature Extraction (page: 85)
        - NLTK là thư viện hỗ trợ phân loại, tạo từ gốc, gắn thẻ, phân tích cú pháp, lập luận ngữ nghĩa, mã hóa 
        - Các thư viện hỗ trợ NLP phổ biến
            + Để thực hiện công việc liên quan đến phân loại/ phân cụm văn bản thì cần phải tiền xử lý với các công việc:
                Bước 1: Tokenizer
                Bước 2: StopwordRemover
                Bước 3: nGram
                Bước 4: TF-IDF
                Bước 5: CountVectorizer
                ...
        - Các công việc nên thực hiện
            Bước 1: Tiền xử lý dữ liệu thô
                - Chuyển text về chữ thường
                - Loại bỏ các kỹ tự đặc biệt nếu có
                - Thay thế những emojicon bằng text tương ứng
                - Thay thế bằng teencode bằng text tương ứng
                - Thay thế bằng punctuation và number bằng khoảng trắng
                - Thay thế bằng các từ sai chính tả bằng khoảng trắng
                - Thay thế 1 loạt khoảng trắng thành 1 khoảng trắng
            Bước 2: Chuẩn hóa unicode tiếng việt
            Bước 3: Tokenizer văn bản tiếng việt bằng thư viện underthesea (có xử lý ghép từ "Không")
            Bước 4: Xóa các stopword tiếng việt
    8.2. Feature transform (page: 95)
    8.3. Feature Selection (page: 114)
        - LASSO (Toán tử thu nhỏ và chọn lọc tối thiểu) là một phương pháp hồi quy liên quan đến việc xử phạt kích thước tuyệt đối của các hệ số hồi quy.
            - Bằng cách xử phạt (hoặc tương đương ràng buộc tổng các giá trị tuyệt đối của các ước tính), 
            bạn sẽ gặp phải tình huống trong đó một số ước tính tham số có thể chính xác bằng không. Hình phạt được áp dụng càng lớn, ước tính càng bị thu hẹp về không.
            - Điều này thuận tiện khi chúng ta muốn một số tính năng tự động / lựa chọn biến hoặc khi xử lý các yếu tố dự đoán tương quan cao, 
            trong đó hồi quy tiêu chuẩn thường sẽ có các hệ số hồi quy 'quá lớn'.
        - RandomForest
            - https://github.com/runawayhorse001/AutoFeatures
    8.4. Unbalanced data: Undersampling
    
          
'''


from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import QuantileDiscretizer, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.sql.functions import col

'''
    parameter:
        - df: thể pyspark của bảng dữ liệu gồm 2 feature: ID và sentence
        - inputCol: sentence
        - outputCol: word
        - outputCol_stop_word: removeded
        - outputCol_ngrams: ngrams
        - inputCol_idf = "rawFeatures"
        - outputCol_binarizer="binarized_feature"
    return:
        - 
'''
# 3.1. Thực hiện các bước phân cụm/ gán nhãn feature_transform (page: 95)
def feature_transform(df, threshold=0.5 ,inputCol ="sentence", outputCol="word", outputCol_stop_word = "removeded", outputCol_ngrams="ngrams", inputCol_idf = "rawFeatures", outputCol_idf="features", outputCol_binarizer="binarized_feature"):
    # Bước 1: Tokenizer- phân tách từ
    tokenizer = Tokenizer(inputCol=inputCol, outputCol=outputCol)
    regexTokenizer = RegexTokenizer(inputCol=inputCol, outputCol=outputCol, pattern="\\W")
    # alternatively, pattern="\\w+", gaps(False)
    countTokens = udf(lambda words: len(words), IntegerType())
    tokenized = tokenizer.transform(df)
    tokenized.select(inputCol, outputCol)\
    .withColumn("tokens", countTokens(col(outputCol))).show(truncate=False)
    regexTokenized = regexTokenizer.transform(df)
    regexTokenized.select(inputCol, outputCol) \
    .withColumn("tokens", countTokens(col(outputCol))).show(truncate=False)
    # Bước 2: Thực hiện stopword
    remover = StopWordsRemover(inputCol=outputCol, outputCol=outputCol_stop_word)
    #remover.transform(df).show(truncate=False)
    # Bước 3: NGram 
    ngram = NGram(n=2, inputCol=outputCol, outputCol=outputCol_ngrams)
    idf = IDF(inputCol=inputCol_idf, outputCol=outputCol_idf)
    pipeline = Pipeline(stages=[tokenizer, ngram])
    model = pipeline.fit(df)
    model.transform(df).show(truncate=False)
    # Bước 4: Binarizer
    binarizer = Binarizer(threshold=threshold, inputCol=outputCol_idf, outputCol=outputCol_binarizer)
    binarizedDataFrame = binarizer.transform(df)
    return binarizedDataFrame

# 3.2. Bucketizer: Phân loại nhãn cho dữ liệu số (page: 98)
def Bucketizer_feature(df, inputCol, outputCol):
    
    splits = [-float("inf"),3, 10,float("inf")]
    result_bucketizer = Bucketizer(splits=splits, inputCol=inputCol,outputCol=outputCol).transform(df)
    
    return result_bucketizer

# Calculate undersampling Ratio

import math
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


Linear_Regression = '''
    page: 128
'''

# Convert the data to dense vector (features and label)

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

# getdummy - Supervised learning version:
def get_dummy_supervised_learning(df, indexCol, categoricalCols, continuousCols, labelCol):
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))for c in categoricalCols ]
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),outputCol="{0}_encoded".format(indexer.getOutputCol()))for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]+ continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders +[assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    data = data.withColumn('label',col(labelCol))
    if indexCol:
        return data.select(indexCol,'features','label')
    else:
        return data.select('features','label')
    
# getdummy - Unsupervised learning version:

def get_dummy_unsupervised_learning(df,indexCol,categoricalCols,continuousCols):
    '''
    Get dummy variables and concat with continuous variables for unsupervised learning.
    :param df: the dataframe
    :param categoricalCols: the name list of the categorical data
    :param continuousCols: the name list of the numerical data
    :return k: feature matrix
'''
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categoricalCols ]
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
    outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]+ continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders +[assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    if indexCol:
        return data.select(indexCol,'features')
    else:
        return data.select('features')
    

def get_dummy_one_feature_unsupervised_learning(df, indexCol, categoricalCols, continuousCols, labelCol, dropLast=False):
    '''
        Get dummy variables and concat with continuous variables for ml
        ˓→modeling.
        :param df: the dataframe
        :param categoricalCols: the name list of the categorical data
        :param continuousCols: the name list of the numerical data
        :param labelCol: the name of label column
        :param dropLast: the flag of drop last column
        :return: feature matrix
        :author: Wenqiang Feng
        :email: von198@gmail.com
        >>> df = spark.createDataFrame([
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "a"),
        (4, "a"),
        (5, "c")
        ], ["id", "category"])
        >>> indexCol = 'id'
        >>> categoricalCols = ['category']
        >>> continuousCols = []
        >>> labelCol = []
        >>> mat = get_dummy(df,indexCol,categoricalCols,continuousCols,
        ˓→labelCol)
        >>> mat.show()
        >>>
        +---+-------------+
        | id| features|
        +---+-------------+
        | 0|[1.0,0.0,0.0]|
        | 1|[0.0,0.0,1.0]|
        | 2|[0.0,1.0,0.0]|
        | 3|[1.0,0.0,0.0]|
        | 4|[1.0,0.0,0.0]|
        | 5|[0.0,1.0,0.0]|
        +---+-------------+
    '''
    
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))for c in categoricalCols ]
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),outputCol="{0}_encoded".format(indexer.getOutputCol()),dropLast=dropLast)for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    if indexCol and labelCol:
    # for supervised learning
        data = data.withColumn('label',col(labelCol))
        return data.select(indexCol,'features','label')
    elif not indexCol and labelCol:
    # for supervised learning
        data = data.withColumn('label',col(labelCol))
        return data.select('features','label')
    elif indexCol and not labelCol:
    # for unsupervised learning
        return data.select(indexCol,'features')
    elif not indexCol and not labelCol:
    # for unsupervised learning
        return data.select('features')