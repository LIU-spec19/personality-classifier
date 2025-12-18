// 替换模拟预测为真实API请求
async function generatePrediction(data) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        // 补充关键特征分析
        result.keyFeatures = analyzeKeyFeatures(data);
        return result;
    } catch (error) {
        console.error('预测错误:', error);
        alert('预测失败，请重试！');
        throw error;
    }
}