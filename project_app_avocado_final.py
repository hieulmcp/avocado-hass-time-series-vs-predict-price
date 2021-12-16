# 1. Import library
import lib.lib_summary as pre
import lib.step11ml_utilis_model_design_time_series as ser
import lib.step15_processing_pipeline as pro_pre
import datetime
import pandas as pd
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
########################################################################################################
# STREAMLIT
########################################################################################################
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield
    
##############################################################################################################
# B. Dự báo giá bán
##############################################################################################################
#try:
st.sidebar.markdown("<h1 style='text-align: left; color: Yellow;'>HÃY LỰA CHỌN TÔI</h1>", unsafe_allow_html=True)
menu = ["Summary", "Model/Evaluate predict avocado prices", 'Predict avocado prices', "Time series"]
#choice = st.sidebar.selectbox('Menu',menu)
choice = st.sidebar.radio("Chọn nội dung mà bạn muốn xem ?",("Nhìn chung về bơ hass","Tổng quan và nghiên cứu thị trường", "Model/Evaluate predict avocado prices", 'Predict avocado prices', "Time series", "Kết luận và hướng phát triển dự án"))

if choice == 'Nhìn chung về bơ hass':

    st.markdown("<h1 style='text-align: center; color: Yellow;'>AVOCADO HASS MEXICO</h1>", unsafe_allow_html=True)
    video_file = open('video/video.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.write(" ")

    st.markdown("<h1 style='text-align: center; color: Yellow;'>TỔNG QUAN DỰ ÁN</h1>", unsafe_allow_html=True)
    video_file = open('video/Tong_quan_ve_du_an.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.write(" ")

    st.markdown("<h1 style='text-align: center; color: Yellow;'>HƯỚNG DẪN CHO NGƯỜI DÙNG</h1>", unsafe_allow_html=True)
    video_file = open('video/Huong_dan_su_dung.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.write(" ")
    
    

elif choice == 'Tổng quan và nghiên cứu thị trường':
    
    st.markdown("<h1 style='text-align: center; color: Coral;'>TẦM NHÌN THỰC HIỆN DỰ ÁN BƠ TẠI HOA KỲ</h1>", unsafe_allow_html=True)
    st.image('picture/avocado_face.jpg')
    st.write(" ")
    
    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Thị trường quả bơ của Hoa Kỳ</h3>", unsafe_allow_html=True)
    st.image('picture/market_USD.PNG')
    st.markdown("- Ở Hoa Kỳ, bơ được bán trên thị trường như là một lựa chọn dinh dưỡng sức khỏe và là một nguồn tốt bổ sung dầu không bão hòa đơn (monounsaturated) có lợi. Một quả bơ trung bình chứa khoảng 15% lượng chất béo bão hòa hàng ngày được FDA khuyến nghị sử dụng") 
    st.markdown("- Theo Bộ Nông nghiệp Hoa Kỳ (USDA), tổng giá trị quả bơ nhập khẩu của Hoa Kỳ năm 2021 là 2,35 tỷ USD, giảm 11%, từ 2,64 tỷ USD năm 2020, tuy nhiên lại tăng về khối lượng, Hoa Kỳ đã nhập 1,04 triệu tấn, tăng 15%, từ mức 0,9 triệu tấn năm 2020")
    st.markdown("- Mexico với lợi thế là nguồn cung lớn nhất và cũng là nước có biên giới chung với Hoa Kỳ là nước đứng đầu, chiếm 87% về khối lượng và 88% giá trị tổng kim ngạch nhập khẩu, tăng 17% về khối lượng nhưng giảm 11% về giá trị trong năm 2021, Peru là nhà cung cấp quả bơ lớn thứ 2, chiếm 8% về khối lượng cũng như giá trị, Chile ở vị trí thứ 3, chiếm 3% tổng khối lượng và giá trị.")
    st.markdown("- Mặc dù là nhà nhập khẩu ròng, Hoa Kỳ cũng là quốc gia trồng quả bơ. Quả bơ thương mại của Hoa Kỳ chủ yếu từ 3 bang là California, Florida và Hawaii có nguồn gốc từ Tây Ấn (West Indies), Guatemala, Mexico hoặc các giống lai của chúng, trong đó giống bơ Hass với hàm lượng dầu 18% - 22% là loại tốt nhất, tiếp theo là giống Fuerte (12% - 17%)")
    st.markdown("- Theo số liệu thống kê mới nhất , quả bơ sản xuất của Hoa Kỳ năm 2017 đạt 146.310 tấn (đạt giá trị khoảng 392 triệu USD). Giống Hass có xu hướng chịu đựng tốt và cho năng suất cao. Ở Hoa Kỳ, bơ được bán trên thị trường như là một lựa chọn dinh dưỡng sức khỏe và là một nguồn tốt bổ sung dầu không bão hòa đơn (monounsaturated) có lợi. Một quả bơ trung bình chứa khoảng 15% lượng chất béo bão hòa hàng ngày được FDA khuyến nghị sử dụng")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Mục tiêu của dự án</h3>", unsafe_allow_html=True)
    st.image('picture/Goal.PNG')
    st.write(" ")
    

    st.markdown("<h3 style='text-align: left; color: Aqua;'>3. Vì sao có dự án nào ?</h3>", unsafe_allow_html=True)
    st.image('picture/why.PNG')
    st.write(" ")


    st.markdown("<h3 style='text-align: left; color: Aqua;'>4. Vấn đề hiện tại của doanh nghiệp ?</h3>", unsafe_allow_html=True)
    st.markdown("- Doanh nghiệp chưa có mô hình dự báo giá bơ cho việc mở rộng")
    st.markdown("- Tối ưu sao việc tiếp cận giá bơ tới người tiêu dùng thấp nhất")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>5. Thông tin dữ liệu ?</h3>", unsafe_allow_html=True)
    st.image('picture/dataset.PNG')
    st.write(" ")

elif choice == 'Model/Evaluate predict avocado prices':
    # 1.0. upload data
    dir_file = "data/avocado_model.csv"
    df = pd.read_csv(dir_file)

    #1.1. Chạy model

    #1.1.1. Chọn list dữ liệu cần thực hiện
    lst_k_best_option1 = ['fea_average_price','fea_type_organic', 'fea_large_bags', 'fea_item_4046', 'fea_item_4225', 'fea_small_bags', 'fea_total_Bags', 'fea_item_4770', 'fea_month', 
    'total_volumn_item', 'fea_total_Volume','fea_xlarge_bags' ]
    lst_name = ['fea_type_organic', 'fea_large_bags', 'fea_item_4046',  'fea_item_4225', 'fea_small_bags', 'fea_total_Bags', 'fea_item_4770', 'fea_month', 
    'total_volumn_item', 'fea_total_Volume','fea_xlarge_bags' ]

    #1.1.2. Chọn model best
    data_model_k_best_option1 = df[lst_k_best_option1]


    #Kiểm tra dữ liệu null/ missing/ scaler/ Train test
    check = pre.data_preprocessing(data_model_k_best_option1, y="fea_average_price", task="regression")

    # Xem mung do can bang cua du lieu
    dtf_train, dtf_test = pre.dtf_partitioning(data_model_k_best_option1, y="fea_average_price", test_size=0.3, shuffle=False)

    # Chon bien target và biến input
    X = data_model_k_best_option1.drop(['fea_average_price'], axis=1)
    y = data_model_k_best_option1['fea_average_price']


    # Chia dữ liệu (Data splitting)
    X_train, X_test, y_train, y_test = pre.train_test_split(X, y, random_state=42, test_size=0.3)

    # Chuyển về series
    X_train = X_train[lst_name].values
    y_train = y_train.values

    # Chuyển về series
    X_test = X_test[lst_name].values
    y_test = y_test.values

    # Chay mo hinh
    model_rf = pre.ExtraTreesRegressor()
    model_rf = model_rf.fit(X_train, y_train)


    # tinh RMSE, MAE, R^2
    model = model_rf
    Train_Dataset_RMSE = pre.mean_squared_error(y_true=y_train, y_pred=model.predict(X_train), squared=False)
    Test_Dataset_RMSE = pre.mean_squared_error(y_true=y_test, y_pred=model.predict(X_test), squared=False)
    Train_Dataset_R_square = model.score(X_train, y_train)
    Test_Dataset_R_square = model.score(X_test, y_test)


    y_pred=model.predict(X_train)
    predicted=model.predict(X_test)


    fig1  = pre.evaluate_regr_model(y_train, y_pred, figsize=(25,5))

    fig2 = pre.evaluate_regr_model(y_test, predicted, figsize=(25,5))

    #i = 1
    #print("True:", "{:,.0f}".format(y_test[i]), "--> Pred:", "{:,.0f}".format(predicted[i]))

    #pre.explainer_shap(model, lst_name, X_instance=X_test[i], X_train=None, task="regression", top=10)


    st.markdown("<h1 style='text-align: Center; color: Yellow;'>MÔ HÌNH DỰ ĐOÁN GIÁ BƠ</h1>", unsafe_allow_html=True)
    st.image('picture/model_avocado.png')
    

    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Upload dữ liệu</h3>", unsafe_allow_html=True)
    data_model_k_best_option1
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Kiểm tra dữ liệu</h3>", unsafe_allow_html=True)
    output = st.empty()
    with st_capture(output.code):
        print(pre.data_preprocessing(data_model_k_best_option1, y="fea_average_price", task="regression"))
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>3. Cân bằng dữ liệu train và test sau khi chia</h3>", unsafe_allow_html=True)
    output = st.empty()
    with st_capture(output.code):
        print(pre.dtf_partitioning(data_model_k_best_option1, y="fea_average_price", test_size=0.3, shuffle=False))
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>4. Đánh giá model</h3>", unsafe_allow_html=True)
    output = st.empty()
    with st_capture(output.code):
        print('Train Dataset RMSE: ', pre.mean_squared_error(y_true=y_train, y_pred=model.predict(X_train), squared=False))
        print('Test Dataset RMSE: ', pre.mean_squared_error(y_true=y_test, y_pred=model.predict(X_test), squared=False))
        print('Train Dataset R-square: ', model.score(X_train, y_train))
        print('Test Dataset R-square: ', model.score(X_test, y_test))
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>5. Đánh giá tập train và trực quan hóa</h3>", unsafe_allow_html=True)
    pre.evaluate_regr_model(y_train, y_pred, figsize=(25,5))
    output = st.empty()
    with st_capture(output.code):
        print(pre.evaluate_regr_model(y_train, y_pred, figsize=(25,5)))
    st.write(" ")

    st.pyplot()


    st.markdown("<h3 style='text-align: left; color: Aqua;'>6. Đánh giá tập test và trực quan hóa</h3>", unsafe_allow_html=True)
    pre.evaluate_regr_model(y_test, predicted, figsize=(25,5))
    output = st.empty()
    with st_capture(output.code):
        print(pre.evaluate_regr_model(y_test, predicted, figsize=(25,5)))
    st.write(" ")

    st.pyplot()
    
    #st.markdown("<h3 style='text-align: left; color: Aqua;'>7. Ảnh hưởng các biến</h3>", unsafe_allow_html=True)
    #i = 1
    #print("True:", "{:,.0f}".format(y_test[i]), "--> Pred:", "{:,.0f}".format(predicted[i]))

    #pre.explainer_shap(model, lst_name, X_instance=X_test[i], X_train=None, task="regression", top=10)
    #st.pyplot()
    #st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>7. Kết luận</h3>", unsafe_allow_html=True)
    st.markdown("- Doanh nghiệp có thể sử dụng được model này để dự báo")
    st.write(" ")

elif choice == 'Predict avocado prices':

    st.markdown("<h1 style='text-align: Center; color: Yellow;'>DỰ ĐOÁN GIÁ BƠ THEO BẢNG BÊN DƯỚI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Hướng dẫn sử dụng</h3>", unsafe_allow_html=True)
    st.image('picture/huong_dan_su_dung.PNG')
    st.write(" ")

    # Mở model
    #my_model = Load_Object('model_price.pkl')
    dir_file = "data/avocado_model.csv"
    df = pd.read_csv(dir_file)
    df.head()

    lst_k_best_option1 = ['fea_average_price','fea_type_organic', 'fea_large_bags', 'fea_item_4046', 'fea_item_4225', 'fea_small_bags', 'fea_total_Bags', 'fea_item_4770', 'fea_month', 
    'total_volumn_item', 'fea_total_Volume','fea_xlarge_bags' ]
    lst_name = ['fea_type_organic', 'fea_large_bags', 'fea_item_4046',  'fea_item_4225', 'fea_small_bags', 'fea_total_Bags', 'fea_item_4770', 'fea_month', 
        'total_volumn_item', 'fea_total_Volume','fea_xlarge_bags' ]


    data_model_k_best_option1 = df[lst_k_best_option1]
    X = data_model_k_best_option1.drop(['fea_average_price'], axis=1)
    y = data_model_k_best_option1['fea_average_price']
    # Chia dữ liệu (Data splitting)
    X_train, X_test, y_train, y_test = pre.train_test_split(X, y, random_state=42, test_size=0.3)

    X_train = X_train[lst_name].values
    y_train = y_train.values

    X_test = X_test[lst_name].values
    y_test = y_test.values

    model_rf = pre.ExtraTreesRegressor()
    model_rf = model_rf.fit(X_train, y_train)

    my_model = model_rf
    # Giao dien nguoi dung
    #max_parch = max(df_["fea_item_4046"]) 
    #fea_item_4046_ = st.slider("Mã item 4046 bán trong tuần", 0, 30000000, 1)
    st.sidebar.markdown("<h6 style='text-align: left; color: Aqua;'>CHỌN LOẠI DỰ ĐOÁN</h6>", unsafe_allow_html=True)
    content_ = st.sidebar.radio("Chọn quyền muốn xem ?",('Dự đoán theo file ?', 'Nhập theo nội dung cần chọn ?'))
    
    if content_ == 'Nhập theo nội dung cần chọn ?':
        dir_file = "data/avocado_model.csv"
        df = pd.read_csv(dir_file)
        lst_region = df['region'].unique().tolist()
        lst_type = df['type'].unique().tolist()

        st.sidebar.markdown("<h6 style='text-align: left; color: Aqua;'>THÔNG TIN CẦN NHẬP DỰ ĐOÁN GIÁ BƠ</h6>", unsafe_allow_html=True)
        content_ = st.sidebar.radio("Chọn nội dung mà bạn muốn xem ?",('conventional', 'organic'))

        if content_ == 'conventional':
            fea_type_organic_ = 0
        else:
            fea_type_organic_ = 1

        
        fea_item_4046_ = st.sidebar.number_input("Mã item 4046 bán trong tuần", value = 1)
        fea_item_4225_ = st.sidebar.number_input("Mã item 4225 bán trong tuần", value = 1)
        fea_item_4770_ = st.sidebar.number_input("Mã item 4770 bán trong tuần", value = 1)
        fea_small_bags_ = st.sidebar.number_input("Số lượng túi small bags", value = 1)
        fea_large_bags_ = st.sidebar.number_input("Số lượng túi large bags", value = 1)
        fea_xlarge_bags_ = st.sidebar.number_input("Số lượng túi xlarge bags", value = 1)
        lst_type = [1,2,3,4,5,6,7,8,9,10,11,12]
        #list(1,2,3,4,5,6,7,8,9,10,11,12)
        type_month = st.sidebar.selectbox('Lựa chọn tháng trong năm:', (lst_type))
        # Lấy giá trị cho biến
        fea_type_organic = fea_type_organic_
        fea_item_4046 = fea_item_4046_
        fea_item_4225 = fea_item_4225_
        fea_item_4770 = fea_item_4770_
        total_volumn_item = fea_item_4046 + fea_item_4225 + fea_item_4770
        fea_total_Volume = fea_item_4046 + fea_item_4225 + fea_item_4770
        fea_small_bags = fea_small_bags_
        fea_large_bags = fea_large_bags_
        fea_xlarge_bags =fea_xlarge_bags_
        fea_total_Bags = fea_small_bags+fea_large_bags+fea_xlarge_bags
        fea_month = type_month


        if st.sidebar.button("Hãy nhấn vào tôi đi nào 🤡"):

            # Chuyển đổi dữ liệu qua dataframe
            lst = [fea_type_organic, fea_item_4046, fea_item_4225, fea_item_4770, total_volumn_item, fea_total_Volume, fea_small_bags, fea_large_bags, fea_xlarge_bags, fea_total_Bags, fea_month]
            names = ['fea_type_organic', 'fea_item_4046', 'fea_item_4225', 'fea_item_4770', 'total_volumn_item', 'fea_total_Volume', 'fea_small_bags', 'fea_large_bags', 'fea_xlarge_bags', 'fea_total_Bags', 'fea_month' ]
            df_new = pd.DataFrame (lst)
            X_test = df_new.T
            X_test.columns = names
            st.markdown("<h3 style='text-align: left; color: Aqua;'>2.1. Hiện thị bảng nhập liệu</h3>", unsafe_allow_html=True)
            X_test
            st.write(" ")
            st.markdown("<h3 style='text-align: left; color: Aqua;'>2.2. Dự báo kết quả</h3>", unsafe_allow_html=True)
            # Dự đoán dữ liệu
            predicted = my_model.predict(X_test)
        
            output = st.empty()
            with st_capture(output.code):
                # -> There is a slight trend and it's linear ("additive")
                print("Giá bơ dự đoán theo số liệu trên là: ", round(predicted[0],2))
            st.write(" ")
    elif content_ == 'Dự đoán theo file ?':
        st.markdown("<h3 style='text-align: left; color: Aqua;'>2.1. Chọn đường dẫn file</h3>", unsafe_allow_html=True)
        try:
            uploaded_file = st.file_uploader('Chọn đường dẫn đến tập tin cần dự báo: ', type = ['csv'])
            dir_file = 'data/' + uploaded_file.name
        except Exception as failGeneral:
        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Mô tả:", failGeneral)

        finally:
            print("close")    
        

        if st.button("Hãy nhấn vào tôi đi nào 🤡"):
            final = pro_pre.pre_processing_data(dir_file)
            final = final.reset_index()
            final = final.drop(['index'], axis=1)
            predicted = my_model.predict(final)
            predicted_final = pd.DataFrame(predicted, columns = ['predicted'])
            # compare data
            lst_concat = [predicted_final, final]
            data_model_final = pre.dataframe_concat(lst_concat=lst_concat)
            st.markdown("<h3 style='text-align: left; color: Aqua;'>2.2. Hiện thị kết quả</h3>", unsafe_allow_html=True)
            data_model_final

elif choice == 'Time series':
    
    dir_file = "data/avocado_model.csv"
    df = pd.read_csv(dir_file)
    lst_region = df['region'].unique().tolist()
    lst_type = df['type'].unique().tolist()




    st.markdown("<h1 style='text-align: center; color: Yellow;'>MÔ HÌNH TIME SERIES</h1>", unsafe_allow_html=True)
    #st.image('picture/time_series.jpg')
    st.write(" ")
    content_ = st.sidebar.radio("Chọn nội dung mà bạn muốn xem ?",("Report", "Admin","User"))
    type_predic = st.sidebar.multiselect('Lựa chọn loại bơ bạn muốn dự báo giá:', lst_type, default=['organic'])
    region_predic = st.sidebar.multiselect('Lựa chọn vùng bạn muốn dự báo giá:', lst_region, default=['California'])
    
    #lsts_ = ['Price', 'Volume']
    #selected_variable = st.selectbox('Chọn sản phẩm đi nào bạn',(list(lsts_)))
    selected_variable = st.sidebar.radio("Chọn dự báo về giá hay lượng bạn muốn ?",('Price', 'Volume'))

    # Thêm dữ liệu lsst
    
    #menu1 = ["Custom Model", "Prophet", 'Neural Network','Autoregressive', 'SarimaX', 'Random Walk']
    #choice1 = st.sidebar.selectbox('Model',menu1)
    choice1 = st.sidebar.radio("Chọn model bạn muốn ?",("Custom Model", "Prophet", 'Autoregressive', 'SarimaX', 'Random Walk'))
    
    date_ = st.sidebar.date_input('Bạn muộn dự đoán ngày nào ?', datetime.date(2022,3,24)) # value="2022-03-24"
    
    # Lọc dữ liệu conventional
    feature_name = 'type'
    name_type = type_predic
    # Lọc dữ liệu bang California
    name_type1 = region_predic
    feature_name1 = 'region'

    if content_ == "Admin":
    
        if st.sidebar.button("Hãy nhấn vào tôi đi nào 🤡"):
            data_california = df[(df[feature_name].isin(type_predic))&(df[feature_name1].isin(region_predic))]

            ### 1.2. Thực hiện pre-processing dữ liệu
            data_california["fea_date"] = pd.to_datetime(data_california['fea_date'], format='%Y-%m-%d')
            
            # Chọn giá hay lượng
            
            if selected_variable == 'Price':
                choose_amt_price = 'fea_average_price'
            elif selected_variable == 'Volume':
                choose_amt_price = 'fea_total_Volume'
            

            data_processing = data_california.groupby("fea_date")[choose_amt_price].sum().rename(choose_amt_price)


            st.markdown("<h5 style='text-align: left; color: Aqua;'>1. Upload dữ liệu</h5>", unsafe_allow_html=True)
            data_processing
            st.write(" ")

            end = date_

            st.markdown("<h5 style='text-align: left; color: Aqua;'>2. Target Variable</h5>", unsafe_allow_html=True)
            output = st.empty()
            with st_capture(output.code):
                print("population --> len:", len(data_processing), "| mean:", round(data_processing.mean()), " | std:", round(data_processing.std()))
                w = 30
                print("moving --> len:", w, " | mean:", round(data_processing.ewm(span=w).mean()[-1]), " | std:", round(data_processing.ewm(span=w).std()[-1]))
            st.write(" ")
            ser.plot_ts(data_processing, plot_ma=True, plot_intervals=True, window=w, figsize=(15,5))
            st.pyplot()
            st.write(" ")
            ser.fit_trend(data_processing, degree=1, plot=True, figsize=(15,5))
            st.pyplot()
            st.write(" ")
            trend, line = ser.fit_trend(data_processing, degree=1, plot=True, figsize=(15,5))
            output = st.empty()
            with st_capture(output.code):
                # -> There is a slight trend and it's linear ("additive")
                print("constant:", round(line[-1],2), "| slope:", round(line[0],2))
            st.write(" ")



            st.markdown("<h5 style='text-align: left; color: Aqua;'>3. Level</h5>", unsafe_allow_html=True)
            res_sup = ser.resistence_support(data_processing, window=30, trend=False, plot=True, figsize=(15,5))
            st.pyplot()
            st.write(" ")


            st.markdown("<h5 style='text-align: left; color: Aqua;'>4. Stationarity</h5>", unsafe_allow_html=True)
            ser.test_stationarity_acf_pacf(data_processing, sample=0.20, maxlag=w, figsize=(15,5))
            st.pyplot()
            st.write(" ")
            ser.test_stationarity_acf_pacf(ser.diff_ts(data_processing, order=1), sample=0.20, maxlag=30, figsize=(15,5))
            st.pyplot()
            st.write(" ")

            st.markdown("<h5 style='text-align: left; color: Aqua;'>5. Seasonality</h5>", unsafe_allow_html=True)
            dic_decomposed = ser.decompose_ts(data_processing, s=6, figsize=(15,10))
            st.pyplot()
            st.write(" ")
            #### -> Using weekly seasonality there are smaller residuals
            s = 6
            st.write(" ")

            # ### 1.4. - Preprocessing

            #### 1.4.2. Partitioning
            st.markdown("<h5 style='text-align: left; color: Aqua;'>6. Partitioning</h5>", unsafe_allow_html=True)
            ts_train, ts_test = ser.split_train_test(data_processing, exog=None, test="2017-08-27", plot=True, figsize=(15,5))
            
            output = st.empty()
            with st_capture(output.code):
                # -> There is a slight trend and it's linear ("additive")
                print("train:", len(ts_train), "obs  |  test:", len(ts_test), "obs")
            st.pyplot()
            st.write(" ")


            

            if choice1 == 'Custom Model':
                ### 1.9 - Model Desing & Testing (Custom Model)

                st.markdown("<h5 style='text-align: left; color: Aqua;'>7. Model Desing & Testing</h5>", unsafe_allow_html=True)
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.1. Train / Evaluate</h6>", unsafe_allow_html=True)

                tune = ser.custom_model(ts_train.head(int(0.8*len(ts_train))), pred_ahead=int(0.2*len(ts_train)), 
                                trend=True, seasonality_types=["woy","moy"], 
                                level_window=7, sup_res_windows=(365,365), floor_cap=(True,True), 
                                plot=True, figsize=(15,5))
                st.pyplot()
                st.write(" ")


                trend = True
                seasonality_types = ["woy","moy"]
                level_window = 7
                sup_res_windows = (365,365)
                floor_cap = (True,True)

                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10)))
                dtf = ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10))
                st.pyplot()
                #### 1.9.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.2. Forecast unknown</h6>", unsafe_allow_html=True)

                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5)))
                

                future = ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            
            elif choice1 == 'Prophet':
                ### 1.8. Model Desing & Testing (Prophet)

                #### 1.8.1. Train / Evaluate

                st.markdown("<h5 style='text-align: left; color: Aqua;'>7. Model Desing & Testing</h5>", unsafe_allow_html=True)
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.1. Train / Evaluate</h6>", unsafe_allow_html=True)
                # Create dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), 
                # other additional regressor
                dtf_train = ts_train.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})
                dtf_test = ts_test.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})

                # Create Holidays dataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')
                dtf_holidays = None
                model1 = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10)))
                st.write(" ")

                dtf1, model1 = ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10))
                st.pyplot()

                #### 1.8.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.2. Forecast unknown</h6>", unsafe_allow_html=True)
                dtf = data_processing.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})
                model = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                future = ser.forecast_prophet(dtf, model, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            ### 1.6. - Model Desing & Testing (Autoregressive)
            elif choice1 == 'Autoregressive':
                #### 1.6.1 Exponential Smoothing
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.1. Tuning</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5)))
                st.write(" ")
                res = ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5))
                st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.2. Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10)))
                st.write(" ")
                dtf, model = ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10))
                st.pyplot()
                # Forecast unknown
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.3. Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64))
                    print(ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5)))
                st.write(" ")

                model = ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64)

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5))
                st.pyplot()
            elif choice1 == 'SarimaX':
                # Tuning (this takes a while)

                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.1. Tuning</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5)))
                st.write(" ")
                res = ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5))
                st.pyplot()


                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.2. Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10)))
                st.write(" ")
                # Train/Test
                dtf, model = ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10))
                st.pyplot()


                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.3. Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit())
                    print(ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5)))
                st.write(" ")

                # Forecast unknown
                model = ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit()

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.5 - Baseline (Random Walk)
            elif choice1 == 'Random Walk':
                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.1. Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10)))
                st.write(" ")
                # Train/Test
                dtf = ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10))
                st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>7.2. Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5)))
                    
                st.write(" ")

                # Forecast unknown
                future = ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5))

                st.pyplot()


    elif content_ == "User":
        if st.sidebar.button("Hãy nhấn vào tôi đi nào 🤡"):
            data_california = df[(df[feature_name].isin(type_predic))&(df[feature_name1].isin(region_predic))]
            ### 1.2. Thực hiện pre-processing dữ liệu
            data_california["fea_date"] = pd.to_datetime(data_california['fea_date'], format='%Y-%m-%d')
            # Chọn giá hay lượng
            if selected_variable == 'Price':
                choose_amt_price = 'fea_average_price'
            elif selected_variable == 'Volume':
                choose_amt_price = 'fea_total_Volume'
            data_processing = data_california.groupby("fea_date")[choose_amt_price].sum().rename(choose_amt_price)
            end = date_
            trend, line = ser.fit_trend(data_processing, degree=1, plot=True, figsize=(15,5))
            res_sup = ser.resistence_support(data_processing, window=30, trend=False, plot=True, figsize=(15,5))
            dic_decomposed = ser.decompose_ts(data_processing, s=6, figsize=(15,10))
            s = 6
            ts_train, ts_test = ser.split_train_test(data_processing, exog=None, test="2017-08-27", plot=True, figsize=(15,5))

            if choice1 == 'Custom Model':
                ### 1.9 - Model Desing & Testing (Custom Model)

                st.markdown("<h5 style='text-align: left; color: Aqua;'>Model Desing & Testing</h5>", unsafe_allow_html=True)
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)

                tune = ser.custom_model(ts_train.head(int(0.8*len(ts_train))), pred_ahead=int(0.2*len(ts_train)), 
                                trend=True, seasonality_types=["woy","moy"], 
                                level_window=7, sup_res_windows=(365,365), floor_cap=(True,True), 
                                plot=True, figsize=(15,5))
                st.pyplot()
                st.write(" ")


                trend = True
                seasonality_types = ["woy","moy"]
                level_window = 7
                sup_res_windows = (365,365)
                floor_cap = (True,True)

                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10)))
                dtf = ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10))
                st.pyplot()
                #### 1.9.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)

                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5)))
                

                future = ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            
            elif choice1 == 'Prophet':
                ### 1.8. Model Desing & Testing (Prophet)

                #### 1.8.1. Train / Evaluate

                st.markdown("<h5 style='text-align: left; color: Aqua;'>Model Desing & Testing</h5>", unsafe_allow_html=True)
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                # Create dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), 
                # other additional regressor
                dtf_train = ts_train.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})
                dtf_test = ts_test.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})

                # Create Holidays dataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')
                dtf_holidays = None
                model1 = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10)))
                st.write(" ")

                dtf1, model1 = ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10))
                st.pyplot()

                #### 1.8.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                dtf = data_processing.reset_index().rename(columns={"fea_date":"ds", '"'+choose_amt_price+'"':"y"})
                model = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                future = ser.forecast_prophet(dtf, model, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            ### 1.6. - Model Desing & Testing (Autoregressive)
            elif choice1 == 'Autoregressive':
                #### 1.6.1 Exponential Smoothing
                # Tuning
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5)))
                st.write(" ")
                res = ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5))
                st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10)))
                st.write(" ")
                dtf, model = ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10))
                st.pyplot()
                # Forecast unknown
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64))
                    print(ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5)))
                st.write(" ")

                model = ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64)

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5))
                st.pyplot()
            elif choice1 == 'SarimaX':
                # Tuning (this takes a while)

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5)))
                st.write(" ")
                res = ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5))
                st.pyplot()


                st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10)))
                st.write(" ")
                # Train/Test
                dtf, model = ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10))
                st.pyplot()


                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit())
                    print(ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5)))
                st.write(" ")

                # Forecast unknown
                model = ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit()

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.5 - Baseline (Random Walk)
            elif choice1 == 'Random Walk':
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10)))
                st.write(" ")
                # Train/Test
                dtf = ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10))
                st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                output = st.empty()
                with st_capture(output.code):
                    # -> There is a slight trend and it's linear ("additive")
                    print(ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5)))
                    
                st.write(" ")

                # Forecast unknown
                future = ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5))

                st.pyplot()


    elif content_ == "Report":
        if st.sidebar.button("Hãy nhấn vào tôi đi nào 🤡"):
            st.markdown("<h4 style='text-align: left; color: Aqua;'>A. Price</h4>", unsafe_allow_html=True)
            data_california = df[(df[feature_name].isin(type_predic))&(df[feature_name1].isin(region_predic))]
            ### 1.2. Thực hiện pre-processing dữ liệu
            data_california["fea_date"] = pd.to_datetime(data_california['fea_date'], format='%Y-%m-%d')
            # Chọn giá hay lượng
            data_processing = data_california.groupby("fea_date")['fea_average_price'].sum().rename('fea_average_price')
            end = date_
            trend, line = ser.fit_trend(data_processing, degree=1, plot=True, figsize=(15,5))
            res_sup = ser.resistence_support(data_processing, window=30, trend=False, plot=True, figsize=(15,5))
            dic_decomposed = ser.decompose_ts(data_processing, s=6, figsize=(15,10))
            s = 6
            ts_train, ts_test = ser.split_train_test(data_processing, exog=None, test="2017-08-27", plot=True, figsize=(15,5))

            if choice1 == 'Custom Model':
                ### 1.9 - Model Desing & Testing (Custom Model)
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)

                tune = ser.custom_model(ts_train.head(int(0.8*len(ts_train))), pred_ahead=int(0.2*len(ts_train)), 
                                trend=True, seasonality_types=["woy","moy"], 
                                level_window=7, sup_res_windows=(365,365), floor_cap=(True,True), 
                                plot=True, figsize=(15,5))
                #st.pyplot()
                st.write(" ")


                trend = True
                seasonality_types = ["woy","moy"]
                level_window = 7
                sup_res_windows = (365,365)
                floor_cap = (True,True)
                dtf = ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10))
                #st.pyplot()
                #### 1.9.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)         

                future = ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            
            elif choice1 == 'Prophet':
                ### 1.8. Model Desing & Testing (Prophet)
                #### 1.8.1. Train / Evaluate
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                # Create dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), 
                # other additional regressor
                dtf_train = ts_train.reset_index().rename(columns={"fea_date":"ds", "fea_average_price":"y"})
                dtf_test = ts_test.reset_index().rename(columns={"fea_date":"ds", "fea_average_price":"y"})

                # Create Holidays dataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')
                dtf_holidays = None
                model1 = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                dtf1, model1 = ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10))
                #st.pyplot()

                #### 1.8.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                dtf = data_processing.reset_index().rename(columns={"fea_date":"ds", "fea_average_price":"y"})
                model = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                future = ser.forecast_prophet(dtf, model, end=end, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.6. - Model Desing & Testing (Autoregressive)
            elif choice1 == 'Autoregressive':
                #### 1.6.1 Exponential Smoothing
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                res = ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5))
                #st.pyplot()

                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                dtf, model = ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10))
                #st.pyplot()
                # Forecast unknown
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                model = ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64)

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5))
                st.pyplot()
            elif choice1 == 'SarimaX':
                # Tuning (this takes a while)
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                res = ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5))
                #st.pyplot()


                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                # Train/Test
                dtf, model = ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10))
                #st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                # Forecast unknown
                model = ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit()
                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.5 - Baseline (Random Walk)
            elif choice1 == 'Random Walk':
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                dtf = ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10))
                #st.pyplot()
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                # Forecast unknown
                future = ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5))

                st.pyplot()





            st.markdown("<h4 style='text-align: left; color: Aqua;'>B. Volume</h4>", unsafe_allow_html=True)
            data_california = df[(df[feature_name].isin(type_predic))&(df[feature_name1].isin(region_predic))]
            ### 1.2. Thực hiện pre-processing dữ liệu
            data_california["fea_date"] = pd.to_datetime(data_california['fea_date'], format='%Y-%m-%d')
            data_processing = data_california.groupby("fea_date")['fea_total_Volume'].sum().rename('fea_total_Volume')
            end = date_
            trend, line = ser.fit_trend(data_processing, degree=1, plot=True, figsize=(15,5))
            res_sup = ser.resistence_support(data_processing, window=30, trend=False, plot=True, figsize=(15,5))
            dic_decomposed = ser.decompose_ts(data_processing, s=6, figsize=(15,10))
            s = 6
            ts_train, ts_test = ser.split_train_test(data_processing, exog=None, test="2017-08-27", plot=True, figsize=(15,5))

            if choice1 == 'Custom Model':
                ### 1.9 - Model Desing & Testing (Custom Model)

                #st.markdown("<h5 style='text-align: left; color: Aqua;'>Model Desing & Testing</h5>", unsafe_allow_html=True)
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)

                tune = ser.custom_model(ts_train.head(int(0.8*len(ts_train))), pred_ahead=int(0.2*len(ts_train)), 
                                trend=True, seasonality_types=["woy","moy"], 
                                level_window=7, sup_res_windows=(365,365), floor_cap=(True,True), 
                                plot=True, figsize=(15,5))
                #st.pyplot()
                st.write(" ")


                trend = True
                seasonality_types = ["woy","moy"]
                level_window = 7
                sup_res_windows = (365,365)
                floor_cap = (True,True)
                dtf = ser.fit_custom_model(ts_train, ts_test, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                conf=0.1, figsize=(15,10))
                #st.pyplot()
                #### 1.9.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)         

                future = ser.forecast_custom_model(data_processing, trend, seasonality_types, level_window, sup_res_windows, floor_cap,
                                        conf=0.3, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            
            elif choice1 == 'Prophet':
                ### 1.8. Model Desing & Testing (Prophet)

                #### 1.8.1. Train / Evaluate
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                # Create dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), 
                # other additional regressor
                dtf_train = ts_train.reset_index().rename(columns={"fea_date":"ds", "fea_total_Volume":"y"})
                dtf_test = ts_test.reset_index().rename(columns={"fea_date":"ds", "fea_total_Volume":"y"})

                # Create Holidays dataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')
                dtf_holidays = None
                model1 = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                dtf1, model1 = ser.fit_prophet(dtf_train, dtf_test, model=model1, figsize=(15,10))
                #st.pyplot()

                #### 1.8.2. Forecast unknown 
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                dtf = data_processing.reset_index().rename(columns={"fea_date":"ds", "fea_total_Volume":"y"})
                model = ser.Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False,
                        holidays=dtf_holidays, interval_width=0.80)

                future = ser.forecast_prophet(dtf, model, end=end, zoom=30, figsize=(15,5))
                st.pyplot()
            ### 1.7. Model Desing & Testing (Neural Network)
            elif choice1 == 'Neural Network':
                #### 1.7.1. Train/test
                # I will try to expand the memory to 1y, losing 365 days of training. This takes a while.
                s = 136
                n_features = 1
                model = ser.models.Sequential()
                model.add( ser.layers.LSTM(input_shape=(s,n_features), units=50, activation='relu', return_sequences=True) )
                model.add( ser.layers.Dropout(0.2) )
                model.add( ser.layers.LSTM(units=50, activation='relu', return_sequences=False) )
                model.add( ser.layers.Dense(1) )
                model.compile(optimizer='adam', loss='mean_absolute_error')
                model.summary()
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                dtf, model = ser.fit_lstm(ts_train, ts_test, model, exog=None, s=s, epochs=100, conf=0.20, figsize=(15,10))
                #st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                #### 1.7.2. Forecast unknown 
                future = ser.forecast_lstm(data_processing, model, conf=0.20, end=end, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.6. - Model Desing & Testing (Autoregressive)
            elif choice1 == 'Autoregressive':
                #### 1.6.1 Exponential Smoothing
                # Tuning
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                res = ser.tune_expsmooth_model(ts_train, s=s, val_size=0.2, scoring=pre.metrics.mean_absolute_error, top=4, figsize=(15,5))
                #st.pyplot()

                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                dtf, model = ser.fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=s,
                                factors=(None,None,None), conf=0.10, figsize=(15,10))
                #st.pyplot()
                # Forecast unknown
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                model = ser.smt.ExponentialSmoothing(data_processing, trend="additive", damped=False, 
                                        seasonal="multiplicative", seasonal_periods=s).fit(0.64)

                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.30, zoom=30, figsize=(15,5))
                st.pyplot()
            elif choice1 == 'SarimaX':
                # Tuning (this takes a while)
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Tuning</h6>", unsafe_allow_html=True)
                res = ser.tune_arima_model(ts_train, s=s, val_size=0.2, max_order=(1,1,1), seasonal_order=(1,0,1),
                                        scoring=ser.metrics.mean_absolute_error, top=3, figsize=(15,5))
                #st.pyplot()


                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                # Train/Test
                dtf, model = ser.fit_sarimax(ts_train, ts_test, order=(1,1,1), seasonal_order=(1,0,1), s=s, conf=0.95, figsize=(15,10))
                #st.pyplot()

                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                # Forecast unknown
                model = ser.smt.SARIMAX(data_processing, order=(1,1,1), seasonal_order=(1,0,1,s), exog=None).fit()
                future = ser.forecast_autoregressive(data_processing, model, end=end, conf=0.95, zoom=30, figsize=(15,5))
                st.pyplot()

            ### 1.5 - Baseline (Random Walk)
            elif choice1 == 'Random Walk':
                #st.markdown("<h6 style='text-align: left; color: Aqua;'>Train / Evaluate</h6>", unsafe_allow_html=True)
                dtf = ser.simulate_rw(ts_train, ts_test, conf=0.10, figsize=(15,10))
                #st.pyplot()
                st.markdown("<h6 style='text-align: left; color: Aqua;'>Forecast unknown</h6>", unsafe_allow_html=True)
                # Forecast unknown
                future = ser.forecast_rw(data_processing, end=end, conf=0.10, zoom=30, figsize=(15,5))

                st.pyplot()

elif choice == 'Kết luận và hướng phát triển dự án':
    st.markdown("<h1 style='text-align: center; color: Coral;'>KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. SWOT</h3>", unsafe_allow_html=True)
    st.write(" ")
    st.image('picture/SWOT.png')
    
    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Tài liệu tham khảo</h3>", unsafe_allow_html=True)
    st.markdown("https://moit.gov.vn/tin-tuc/quoc-te/gioi-thieu-thi-truong-qua-bo-hoa-ky.html")
    st.markdown("https://www.kaggle.com/yassinealouini/avocado-time-series-modeling")
    st.markdown("https://github.com/")
    st.markdown("Book: B. V. Vishwas, Ashish Patel - Hands-on Time Series Analysis With Python_ From Basics To Bleeding Edge Techniques-Apress (2020).pdf")

#except Exception as failGeneral:
#    output = st.empty()
#    with st_capture(output.code):
#        # -> There is a slight trend and it's linear ("additive")
#        print("Fail system, please call developer...", type(failGeneral).__name__)
#       print("Mô tả:", failGeneral)
#    st.write(" ")
    
#finally:
#   output = st.empty()
#    with st_capture(output.code):
#        # -> There is a slight trend and it's linear ("additive")
#        print("")  
#    st.write(" ")
