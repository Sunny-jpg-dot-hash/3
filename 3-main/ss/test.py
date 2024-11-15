import gymnasium as gym
import numpy as np
import os
import highway_env  # 確保已安裝 highway-env 支援 roundabout-v0 環境
import tensorflow as tf

# 設定根目錄路徑
root_path = os.path.abspath(os.path.dirname(__file__))

def load_validation_data():
    # 加載驗證數據
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = valid_dataset['label']
    return valid_data, valid_label

def main():
    total_reward = 0
    
    # 加載模型
    model_path = os.path.join(root_path, 'YOURMODEL.h5')  # 替換為你的模型路徑
    model = tf.keras.models.load_model(model_path)

    # 自訂驗證，不使用 Keras API
    # 建立環境並檢查 roundabout-v0 是否可用
    try:
        env = gym.make('roundabout-v0', render_mode='rgb_array')
    except gym.error.NameNotFound:
        print("環境 'roundabout-v0' 不存在。請確認已正確安裝 highway-env。")
        return None

    # 加載驗證數據
    valid_data, valid_label = load_validation_data()
    correct_predictions = 0
    total_samples = 0

    for _ in range(10):  # 進行 10 輪測試
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            env.render()
            obs = obs.reshape(1, 25)  # 將觀察數據重塑為 (1, 25)
            
            # 預測動作
            logits = model(obs, training=False)  # 獲取原始 logits
            action = np.argmax(logits.numpy())  # 將 logits 轉換為動作

            # 比較預測結果與真實標籤
            true_label = valid_label[total_samples]  # 假設驗證數據與觀察對齊
            if action == true_label:
                correct_predictions += 1

            # 執行動作
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            total_samples += 1

    # 計算模型的準確率
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # 關閉環境
    env.close()

    # 輸出測試準確率和累計獎勵
    print(f"準確率: {accuracy * 100:.2f}%")
    print(f"10 輪後的總獎勵: {total_reward}")
    return total_reward

if __name__ == "__main__":
    main()  # 只執行一次 10 輪測試並結束
