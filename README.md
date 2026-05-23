# Agentic Multi Model AI Analytics Dashboard 🤖📊

## 📋 Overview
Most machine learning systems just predict. **Ours reasons, decides, and explains.**

This repository contains a production-level intelligent platform powered by a **ReAct LLM Agent** (Groq Llama 3.3 70B). Acting as an autonomous system manager, the agent interprets natural language input, automatically routes the problem to the appropriate machine learning "expert" model, and delivers a structured prediction complete with confidence scoring and full AI reasoning transparency.

## 🧠 The Expert Models
The system architecture supports three distinct real-world problems, handled by specialized models:

### 1. Bank Customer Churn Prediction 🏦
* **Expert Model:** Neural Network (MLP)
* **Dataset:** 10,000 bank customers
* **Performance:** F1 Score 0.853 | Accuracy 86.3%
* **Problem Solved:** *Will this customer leave the bank?*

### 2. Diabetes Risk Prediction 🏥
* **Expert Model:** Logistic Regression
* **Dataset:** Pima Indians (768 patients)
* **Problem Solved:** *Is this patient at risk for diabetes?*

### 3. SMS Spam Detection 📧
* **Expert Model:** Naive Bayes (NLP)
* **Dataset:** 5,572 SMS messages
* **Performance:** F1 Score 0.985 | Accuracy 98%+
* **Problem Solved:** *Is this message spam?*

## ⚙️ How It Works (The ReAct Framework)
The platform moves beyond standard forms and static inputs by utilizing a dynamic **Thought → Action → Observation** loop:

1. **User Input:** The user describes the problem in natural language.
2. **Thought:** The LLM Agent analyzes the input to identify the core problem type.
3. **Action:** The Agent calls the correct expert ML model.
4. **Observe:** The Agent reads the prediction result.
5. **Analytics Dashboard:** The UI outputs the Risk Level, Confidence Score, AI Reasoning, and a final Explainable Verdict.

## 🛠️ Tech Stack
* **Language & UI:** Python, Streamlit (Production-level dark UI)
* **LLM & Agent Framework:** Groq (Llama 3.3 70B), LangChain, LangGraph
* **Machine Learning:** Scikit-Learn, TensorFlow/Keras (Neural Networks, Naive Bayes, Logistic Regression)

## 🌟 Key Features
* **Full Agentic AI System:** Automatically picks the right expert for every problem.
* **Explainable AI (XAI):** Full reasoning trace is visible to the user.
* **Natural Language Processing:** Just describe the problem—no rigid forms required.
* **Real-Time Analytics:** Enterprise-grade dashboard featuring Risk Level and Confidence scoring.
* **Cost-Efficient:** Built on a 100% free, open-source stack ($0 execution cost).

---
### 👨‍💻 Project Team
* **Fawad Saqib**
* **Awais Manzoor**
* **Muhammad Haisam**

*Supervised by Sir Shah Zain. Deep gratitude for the continuous guidance, mentorship, and unwavering support throughout this journey.*
