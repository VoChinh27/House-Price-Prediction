import pandas as pd
import joblib
import streamlit as st 
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Tải dữ liệu huấn luyện đã được xử lý trước
train_data = pd.read_csv("updated_X.csv")

# Khởi tạo OneHotEncode
one_hot = OneHotEncoder()
categorical_values = ['OverallQual', 'GarageCars', 'TotRmsAbvGrd', 'Neighborhood', 'FullBath', 'GarageType']
transformer = ColumnTransformer([('one_hot', one_hot, categorical_values)],
                                remainder='passthrough')
transformer.fit(train_data)

# Hàm chuyển đổi dữ liệu đầu vào
def transform_data(transformer, data):
    transform_X = transformer.transform(data).toarray()
    transformed_df = pd.DataFrame(transform_X)
    return transformed_df

# Tải mô hình đã huấn luyện
loaded_model = joblib.load("best_model.sav")

#Chức năng dự đoán giá nhà
def house_price_predictions(input_data):
    transformed_data = transform_data(transformer, input_data)
    predictions = loaded_model.predict(transformed_data)
    return predictions

# Xác định nhãn có ý nghĩa cho các tùy chọn
options_names = {
    'OverallQual': {
        10: 'Rất xuất sắc',
        9: 'Xuất sắc',
        8: 'Rất tốt',
        7: 'Tốt',
        6: 'Trên trung bình',
        5: 'Trung bình',
        4: 'Dưới trung bình',
        3: 'Được',
        2: 'Kém',
        1: 'Rất kém'
},
    'GarageCars': {
        0: '0 Xe',
        1: '1 Xe',
        2: '2 Xe',
        3: '3 xe',
        4: '4 xe'
    },
    'TotRmsAbvGrd': {
        2: '2 Phòng',
        3: '3 Phòng',
        4: '4 Phòng',
        5: '5 Phòng',
        6: '6 Phòng',
        7: '7 Phòng',
        8: '8 Phòng',
        9: '9 Phòng',
        10: '10 Phòng',
        11: '11 Phòng',
        12: '12 Phòng',
        13: '13 Phòng',
        14: '14 Phòng'
    },
    'Neighborhood': {
        'NAmes': 'North Ames',
        'CollgCr': 'College Creek',
        'OldTown': 'Old Town',
        'Edwards': 'Edwards',
        'Somerst': 'Somerset',
        'Gilbert': 'Gilbert',
        'NridgHt': 'Northridge Heights',
        'Sawyer': 'Sawyer',
        'NWAmes': 'Northwest Ames',
        'SawyerW': 'Sawyer West',
        'BrkSide': 'Brookside',
        'Crawfor': 'Crawford',
        'Mitchel': 'Mitchell',
        'NoRidge': 'Northridge',
        'Timber': 'Timberland',
        'IDOTRR': 'Iowa DOT and Rail Road',
        'ClearCr': 'Clear Creek',
        'StoneBr': 'Stone Brook',
        'SWISU': 'South & West of Iowa State University',
        'MeadowV': 'Meadow Village',
        'Blmngtn': 'Bloomington Heights',
        'BrDale': 'Briardale',
        'Veenker': 'Veenker',
        'NPkVill': 'Northpark Villa',
        'Blueste': 'Bluestem'
    },
    'FullBath': {
        0: '0 phòng tắm',
        1: '1 phòng tắm',
        2: '2 phòng tắm',
        3: '3 phòng tắm'
    },
    'GarageType': {
        '2Types': 'Hơn một loại',
        'Attchd': 'Gắn liền',
        'Basment': 'Hầm',
        'BuiltIn': 'Xây dựng sẵn',
        'CarPort': 'Chỗ để xe',
        'Detchd': 'Tách rời',
        'missing': 'Không có gara'
    }
}
# Giao Diện
def main():
    st.markdown(
        """
        <style>
        .stApp {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1>Dự Đoán Giá Nhà</h1>", unsafe_allow_html=True)
    st.write("**By Võ Chính | Văn Vương**  \n[GitHub](https://github.com/VoChinh27) | [FaceBook](https://www.facebook.com/vominhchinh.27/)")
    st.markdown("##### **Chào Mừng Đến Dự Án Dự Đoán Giá Nhà.**")
    
    # Preload the image
    image = Image.open("City_House.png")

    # Display the image with centered alignment
    st.image(image)
    
    st.sidebar.header("House Details")

features = {
    'OverallQual': 'Chất lượng tổng thể',
    'GrLivArea': 'Diện tích sử dụng trên mặt đất (m²)',
    'TotalBsmtSF': 'Tổng diện tích tầng hầm (m²)',
    '2ndFlrSF': 'Diện tích tầng hai (m²)',
    'BsmtFinSF1': 'Diện tích tầng hầm đã hoàn thiện (m²)',
    '1stFlrSF': 'Diện tích tầng một (m²)',
    'GarageCars': 'Số lượng xe trong gara',
    'GarageArea': 'Diện tích gara (m²)',
    'LotArea': 'Diện tích lô đất (m²)',
    'TotRmsAbvGrd': 'Tổng số phòng trên mặt đất',
    'Age': 'Tuổi của ngôi nhà',
    'Neighborhood': 'Khu vực',
    'YearRemodAdd': 'Năm cải tạo',
    'MasVnrArea': 'Diện tích ốp gạch (m²)',
    'BsmtUnfSF': 'Diện tích tầng hầm chưa hoàn thiện (m²)',
    'FullBath': 'Số lượng phòng tắm đầy đủ',
    'LotFrontage': 'Chiều dài mặt tiền lô đất (m)',
    'WoodDeckSF': 'Diện tích sàn gỗ (m²)',
    'GarageYrBlt': 'Năm xây dựng gara',
    'GarageType': 'Loại gara'
}



input_data = {}
for feature, label in features.items():
    if feature == 'OverallQual':
        sorted_options = [10,9,8,7,6,5,4,3,2,1]
        input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted_options, format_func=lambda x: options_names[feature][x])
    elif feature in ['GarageCars', 'TotRmsAbvGrd', 'FullBath']:
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature][x])
    elif feature == 'Neighborhood':
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature][x])
    elif feature == 'GarageType':
            input_data[feature] = st.sidebar.selectbox(f"{label}:", sorted(train_data[feature].unique()), format_func=lambda x: options_names[feature].get(x, x))
    else:
            input_data[feature] = st.sidebar.number_input(f"{label}:", min_value=0.0, value=0.0)
            
input_df = pd.DataFrame([input_data])

if st.sidebar.button("Dự Đoán"):
        predicted_price = house_price_predictions(input_df)
        st.write("### Giá Nhà Ước Tính: $", round(predicted_price[0], 2))
            

if __name__ == "__main__":
    main()

