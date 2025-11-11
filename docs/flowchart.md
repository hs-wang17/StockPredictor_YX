# 📈 StockPredictor Pipeline Flowchart / 预测器流程图

This document visualizes the main workflow of the StockPredictor pipeline.  
本文件展示了 StockPredictor 股票预测的主要工作流程。

```mermaid
flowchart TD
    A[🏁 Start / 开始] --> B[⚙️ Load Config & Setup Logger / 加载配置 & 设置日志]
    B --> C[🗓️ Generate Training & Prediction Periods / 生成训练与预测周期]
    C --> D([🔄 For each period / 遍历每个周期])
    
    D --> E[📥 Load & Preprocess Training Data / 加载并预处理训练数据]
    D --> F[📥 Load & Preprocess Prediction Data / 加载并预处理预测数据]

    E --> G[🛠️ Create Training DataLoader / 创建训练 DataLoader]
    F --> H[🛠️ Create Prediction DataLoader / 创建预测 DataLoader]

    G --> I[🧠 Initialize Model / 初始化模型]
    I --> J[🏋️ Train Model / 训练模型]
    J --> K[💾 Save Model per epoch / 每轮/周期保存模型]
    K --> L[📊 Make Predictions using Trained Model / 使用训练模型进行预测]
    H --> L

    L --> M[📝 Append Predictions to all_predictions_list / 添加预测结果到总列表]

    M --> N([🔁 More periods? / 是否还有周期?])
    N -- Yes / 是 --> D
    N -- No / 否 --> O[📎 Concatenate all period predictions / 拼接所有周期预测结果]
    O --> P[💾 Save Combined Predictions CSV / 保存综合预测 CSV]
    P --> Q[🏁 End / 结束]

````
