# 🧠 Self-Made Reinforcement Learning Algorithms  
🚀 **Custom Implementations of Popular RL Algorithms**  
> This repository contains self-made implementations of key **Reinforcement Learning (RL)** algorithms: **Deep Q-Network (DQN)**, **Proximal Policy Optimization (PPO) Discrete**, and **Q-Learning**. These algorithms are used to train agents to make decisions in environments with varying levels of complexity.

---

## 📝 Algorithms Implemented
### 1. **Deep Q-Network (DQN)**  
A powerful model-free RL algorithm that uses deep neural networks to approximate the Q-value function, allowing the agent to handle complex state spaces.  
- **Environment**: Gym-based environments  
- **Features**:  
  - Experience Replay  
  - Target Network  
  - Q-Function Approximation using neural networks

### 2. **Proximal Policy Optimization (PPO) - Discrete**  
An on-policy RL algorithm that optimizes the policy using a clipped objective, ensuring stable learning by limiting the policy updates.  
- **Environment**: Gym-based environments with discrete action spaces  
- **Features**:  
  - Clipped Surrogate Objective  
  - Generalized Advantage Estimation (GAE)  
  - Actor-Critic method

### 3. **Q-Learning**  
A model-free RL algorithm that directly estimates the action-value function without the need for a model of the environment.  
- **Environment**: Gym-based environments with discrete action spaces  
- **Features**:  
  - Q-value table updates  
  - Exploration via epsilon-greedy strategy

---

## 🛠️ Features
- **Custom implementations** of popular RL algorithms  
- Tested on **Gym environments**  
- **Replay buffers** and **target networks** (for DQN)  
- Supports **exploration-exploitation balance**  
- Easily configurable hyperparameters

---

## 📦 Tech Stack
**Language**: Python
- **NumPy**
- **OpenAI Gym**  
- **TensorFlow / PyTorch** (depending on the algorithm)
