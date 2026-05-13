# 📘 Homework 3: DQN and its variants

**Total: 100%**

---

## 1. 📂 Setup & Reference
*   **Base your work on the DRL in Action (English) GitHub repo:**
    *   🔗 [https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)
*   **Use the updated starter code provided by the instructor as your baseline.**

---

## 2. 🧠 HW3-1: Naive DQN for static mode [30%]
*   ✅ **Run the provided code naive or Experience buffer reply**
* 實作Naive DQN（static mode）, DQN + Replay（static mode）, DQN + Replay（random mode）
*   💬 **Chat with ChatGPT about the code to clarify your understanding**
*   📝 **Submit a short understanding report**
    *   Includes:
    *   Training loss curve:Naive DQN（static mode）, DQN + Replay（static mode）, DQN + Replay（random mode）
    *   訓練期間的策略動畫（Dashboard GIF）:Naive DQN（static mode）, DQN + Replay（static mode）, DQN + Replay（random mode）
    *   量化指標：Naive DQN（static mode）, DQN + Replay（static mode）, DQN + Replay（random mode）
    *   討論:Basic DQN implementation for an easy environment
    *   比較:Experience Replay Buffer
    *   總結:why we need replay buffer

---

## 3. ⚖️ HW3-2: Enhanced DQN Variants for player mode [40%]
**Implement and compare the following:**
*   **Use the DQN + replay buffer（player mode）as baseline**
*   **Double DQN**
*   **Dueling DQN**
*   **Dueling + Double DQN**
*   💡 **Focus on how they improve upon the basic DQN approach**
*   📝 **Submit a short understanding report**
    *   Includes:
    *   Training loss curve:Baseline(DQN + replay buffer（player mode）), Double DQN（player mode）, Dueling DQN（player mode）, Dueling + Double DQN（player mode）
    *   訓練期間的策略動畫（Dashboard GIF）:Baseline(DQN + replay buffer（player mode）), Double DQN（player mode）, Dueling DQN（player mode）, Dueling + Double DQN（player mode）
    *   量化指標：Baseline(DQN + replay buffer（player mode）), Double DQN（player mode）, Dueling DQN（player mode）, Dueling + Double DQN（player mode） 
    *   討論：Focus on how they improve upon the basic DQN approach
    *   比較:BaseLine vs Double DQN vs Dueling DQN vs Dueling + Double DQN
    *   總結:which one is the best and why
    

---

## 4. 🔁 HW3-3: Enhance DQN for random mode WITH Training Tips [30%]
**Convert the DQN model from PyTorch to:**
*   **PyTorch Lightning**
*   **Bonus points for integrating training techniques to improve learning**
    *   (e.g., gradient clipping, learning rate scheduling, etc.)
*   📝 **Submit a short understanding report**
    *   Includes:
    *   Training Tips原理介紹:Gradient Clipping, Learning Rate Scheduling, etc
    *   Training loss curve:訓練技巧的消融研究
    *   訓練期間的策略動畫（Dashboard GIF）:訓練技巧的消融研究
    *   量化指標：訓練技巧的消融研究
    *   討論：Converting the DQN model from PyTorch to PyTorch Lightning
    *   比較:BaseLine(DQN + replay buffer（random mode）) vs DQN + replay buffer（random mode） + Training Tips
    *   總結

---

## 5. 🔁 HW3-4: Enhance DQN for random mode WITH Training Tips [30%]
使用 Rainbow DQN 解 Random Mode GridWorld
*   📝 **Submit a short understanding report**
    *   Includes:
    *   Rainbow DQN介紹：哪些技術，可以學到什麼，如何實作
    *   Training loss curve:BaseLine(DQN + replay buffer（random mode）) vs Rainbow DQN（random mode）
    *   訓練期間的策略動畫（Dashboard GIF）:BaseLine(DQN + replay buffer（random mode）) vs Rainbow DQN（random mode）
    *   量化指標：BaseLine(DQN + replay buffer（random mode）) vs Rainbow DQN（random mode）
    *   討論：Converting the DQN model from PyTorch to PyTorch Lightning
    *   比較:BaseLine(DQN + replay buffer（random mode）) vs Rainbow DQN（random mode）
    *   總結