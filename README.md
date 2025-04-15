

### **Day 12 - AI Agent Playing Games (DQN with CartPole)**  
This is part of my **#100DaysOfAI** challenge.  
On **Day 12**, I implemented a **Deep Q-Network (DQN)** agent that learns to balance a pole on a moving cart in the **CartPole** environment — a classic reinforcement learning problem.

---

### **Goal**  
Train an intelligent agent to **solve the CartPole-v1 environment** by learning optimal actions using **Deep Q-Learning**.

---

### **Technologies Used**

| Tool       | Purpose                                              |
|------------|------------------------------------------------------|
| Python     | Main programming language                            |
| PyTorch    | Building and training the deep Q-network             |
| Gymnasium  | Simulated CartPole environment                       |
| NumPy      | Numerical operations                                 |
| Matplotlib | Visualize training performance (reward vs episodes) |
| VS Code    | Code editor                                          |

---

### **How It Works**

1. **Environment Setup**
   - Used the OpenAI Gym CartPole-v1 environment where the agent learns to balance a pole on a moving cart.

2. **Q-Network Architecture**
   - A simple neural network with two hidden layers using ReLU activations.
   - Takes the environment's state as input and outputs Q-values for each action.

3. **Epsilon-Greedy Policy**
   - Agent starts exploring with random actions (high epsilon).
   - Gradually shifts to exploitation by decaying epsilon after each episode.

4. **Experience Replay**
   - Stores past transitions `(state, action, reward, next_state, done)` in a memory buffer.
   - Samples random batches for training to break correlation between consecutive experiences.

5. **Training**
   - Uses MSE loss to minimize the difference between current Q-values and target Q-values.
   - Target Q-value:  
     ```
     Q_target = reward + γ * max(Q_next) * (1 - done)
     ```

6. **Target Network**
   - A clone of the main Q-network that is updated periodically for stable learning.

7. **Visualization**
   - Plotted total reward per episode to monitor agent's performance over time.

---

### **Highlights**

- Applied **Deep Q-Learning** to a real reinforcement learning task.
- Understood the concepts of **value approximation**, **epsilon-greedy exploration**, and **target networks**.
- Implemented **experience replay** for more stable and sample-efficient learning.
- Watched the agent **progressively get better** at balancing the pole.

---

### **What I Learned**

- How **DQN** works and why it improves over basic Q-learning.
- Importance of **decaying epsilon** for exploration vs exploitation.
- How **neural networks approximate Q-values**.
- Why maintaining a **target network** helps prevent instability.
- How to build and train an **AI agent that learns from interaction** with the environment.

---
