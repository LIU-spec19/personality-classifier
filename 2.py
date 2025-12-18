# 后端示例代码（Flask）
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# 加载模型和依赖组件
model = joblib.load('stacking_personality_model.joblib')#训练好的模型
scaler = joblib.load('scaler.joblib')#标准化器
le = joblib.load('label_encoder.joblib')#标签编码器
poly = joblib.load('poly_features.joblib')#多项式特征生成器

@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端传入的特征数据
    data = request.json
    
    # 转换为DataFrame（与训练时格式一致）
    df = pd.DataFrame({
        'Time_spent_Alone': [data['timeAlone']],
        'Social_event_attendance': [data['socialAttendance']],
        'Going_outside': [data['goingOutside']],
        'Friends_circle_size': [data['friendsCircle']],
        'Post_frequency': [data['postFrequency']],
        'Stage_fear_Yes': [1 if data['stageFear'] == 'Yes' else 0],
        'Drained_after_socializing_Yes': [1 if data['drainedAfterSocial'] == 'Yes' else 0]
    })
    
    # 复现特征工程（与训练时一致）
    # 1. 衍生特征
    df['Alone_to_Social_Ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['Social_Comfort_Index'] = (df['Friends_circle_size'] + df['Post_frequency'] - df['Stage_fear_Yes']) / 3
    df['Social_Overload'] = df['Drained_after_socializing_Yes'] * df['Social_event_attendance']
    
    # 2. 独处时间分桶（需与训练时一致）
    df['Time_spent_Alone_Binned'] = pd.qcut(df['Time_spent_Alone'], q=3, labels=['Low', 'Medium', 'High'])
    df = pd.get_dummies(df, columns=['Time_spent_Alone_Binned'], drop_first=True)
    
    # 3. 多项式特征
    poly_features = poly.transform(df[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
    poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
    df[poly_feature_names] = poly_features
    
    # 4. 标准化
    df_scaled = scaler.transform(df)
    
    # 模型预测
    pred_proba = model.predict_proba(df_scaled)[0]
    pred_label = le.inverse_transform([model.predict(df_scaled)[0]])[0]
    
    # 返回结果
    return jsonify({
        'type': pred_label,
        'confidence': round(max(pred_proba) * 100, 1),
        'probabilities': {
            le.classes_[0]: round(pred_proba[0] * 100, 1),
            le.classes_[1]: round(pred_proba[1] * 100, 1)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)