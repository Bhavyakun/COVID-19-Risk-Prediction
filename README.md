# Reinforcement Learning for COVID-19 Risk Prediction

## Project Overview
This project leverages Reinforcement Learning (RL) to predict and manage COVID-19 infection risk based on various factors such as travel history, age, symptoms, and contact history. The goal is to develop an intelligent agent that dynamically adapts to new data and guidelines, ensuring timely and accurate risk assessments.

## Author
**Bhavya Sharma**  
MPS in Data Analytics, Pennsylvania State University  
Email: bks5909@psu.edu

---

## Research/Business Goal
The objective of this project is to build an RL-based model to classify individuals into different COVID-19 risk categories based on their health and travel data. The system aims to assist in early detection and response to potential outbreaks by recommending preventive actions.

## Definition of RL Problem
### **Agent**  
The agent is a predictive model that assigns risk levels based on an individual’s health profile and travel history.

### **Environment**  
The environment consists of a dataset containing individual attributes related to COVID-19, including age, symptoms, and travel data. It evolves as new data and health guidelines emerge.

### **States**  
Each state represents an individual's health profile with the following attributes:
- **Country**: List of visited countries
- **Age**: Categorized into age groups
- **Symptoms**: Fever, cough, sore throat, breathing difficulty, etc.
- **Severity**: Mild, moderate, severe
- **Contact history**: Interaction with a known COVID-19 patient

### **Actions**  
The agent can classify individuals into four risk levels:
- **High Risk**
- **Moderate Risk**
- **Low Risk**
- **No Risk**

### **Reward Function**  
- **Positive Reward**: Given for correct predictions (true positives and true negatives).
- **Negative Reward**: Given for incorrect predictions (false positives and false negatives).

The agent aims to maximize its cumulative reward by improving classification accuracy over time.

---

## Mathematical Foundations
The project is formulated as a Markov Decision Process (MDP):
- **State (S)**: Set of possible individual health profiles
- **Action (A)**: Possible risk classifications
- **Transition Probability (P)**: Probability of transitioning between states based on actions
- **Reward (R)**: Feedback on action accuracy
- **Discount Factor (γ)**: Controls the importance of future rewards

The value function is computed using Bellman Equations:
\[V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]\]

Q-values are updated using the **Q-learning algorithm**:
\[ Q(s,a) \leftarrow Q(s,a) + \alpha [ R + \gamma \max_{a'} Q(s',a') - Q(s,a) ] \]
where \( \alpha \) is the learning rate.

---

## Q-Learning Implementation
### **Algorithm Steps**
1. **Initialize Q-table** with zeros/random values.
2. **For each episode**:
   - Set initial state \( s \).
   - Select an action \( a \) using an epsilon-greedy strategy.
   - Execute action and observe new state \( s' \) and reward \( R \).
   - Update Q-value:
     \[ Q(s,a) \leftarrow Q(s,a) + \alpha [ R + \gamma \max_{a'} Q(s',a') - Q(s,a) ] \]
   - Transition to \( s' \) and repeat until termination.

### **Hyperparameters**
- **Learning Rate (\( \alpha \))**: 0.1
- **Discount Factor (\( \gamma \))**: 0.99
- **Exploration Rate (\( \epsilon \))**: Initially 1.0, decays to 0.01

---

## Dataset
The dataset used in this project is **Covid-19 Dataset.csv**, which contains statistics such as:
- **Confirmed Cases**
- **Active Cases**
- **Deaths per 100 Cases**
- **Recovered per 100 Cases**
- **New Cases**
- **Travel and Contact History**

Data preprocessing includes standardization, feature engineering, and state representation.

---

## Model Evaluation
### **Performance Metrics**
The model is evaluated using:
- **Accuracy**: Correct classifications / total classifications
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall (Sensitivity)**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Measures model’s ability to distinguish between risk levels
- **Cumulative Reward**: Total reward earned over episodes

---

## Results
- The agent was trained for **1000 episodes**.
- The **exploration rate** decayed from 1.0 to 0.01 within the first few episodes.
- The model consistently reached a **score of 185**, suggesting a fixed episode length.
- The rapid decrease in exploration rate indicates **quick adaptation** to the environment.

### **Observations**
- The agent efficiently classifies individuals based on risk levels.
- The reward structure and environment design may need tuning to ensure continued learning.
- Policy decisions can guide risk management, recommending actions like **quarantine, increased testing, and public awareness campaigns**.

---

## Recommendations
- **Optimize Reward Structure**: Ensure the model continues improving beyond early training.
- **Refine State Representations**: Consider additional health indicators or geographic trends.
- **Expand Action Space**: Include interventions beyond risk classification (e.g., hospital recommendations).
- **Monitor Long-Term Trends**: Track evolving pandemic conditions and update policies accordingly.

---

## Conclusion
This project successfully implements an RL-based COVID-19 risk prediction model using Q-learning. The model demonstrates **efficient learning**, **rapid convergence**, and **useful risk assessment insights**. Future enhancements can improve decision-making capabilities and expand its applications to other epidemiological scenarios.

---

## Installation & Usage
### **Prerequisites**
- Python 3.8+
- NumPy
- Pandas
- OpenAI Gym (optional for testing in simulated environments)
- Matplotlib (for visualization)

### **Installation**
```bash
pip install numpy pandas matplotlib gym
```

### **Run the Model**
Execute the Jupyter Notebook:
```bash
jupyter notebook RL\ Final\ code.ipynb
```
Alternatively, run the Python script:
```bash
python q_learning_covid.py
```

### **Customize Parameters**
Modify hyperparameters in the script for tuning:
```python
alpha = 0.1  # Learning Rate
gamma = 0.99  # Discount Factor
epsilon = 1.0  # Exploration Rate
epsilon_decay = 0.995
```

---

## Acknowledgments
This project was developed as part of the **MPS in Data Analytics** program at **Pennsylvania State University**. Special thanks to faculty members and peers for guidance and feedback.

---

## License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute.

