import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from ml_tools import ALL_TOOLS

# Load environment variables (API Key)
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.1, 
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are an intelligent data science assistant named "ML Agent". You orchestrate three machine learning tools:

AVAILABLE MODELS:
1. predict_churn -- Neural Network (MLP) for Bank Customer Churn
   Features (11): CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard(0/1), IsActiveMember(0/1), EstimatedSalary, Geography_Germany(0/1), Geography_Spain(0/1), Gender_Male(0/1)
   
2. predict_diabetes -- Logistic Regression for Diabetes
   Features (8): Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

3. detect_spam -- SVM for SMS/Email Spam Detection
   Input: raw text message string

CRITICAL RULES:
1. AUTO-DETECT which model to use based on the user's message. NEVER ask the user to select a model.
   - If they mention bank, customer, churn, credit, salary, leaving -> predict_churn
   - If they mention diabetes, glucose, blood sugar, BMI, insulin -> predict_diabetes
   - If they mention spam, message, email, SMS, text, suspicious, scam -> detect_spam

2. NEVER REFUSE A PREDICTION: If the user provides partial or insufficient data, DO NOT ask for more information. IMMEDIATELY fill missing values with sensible median/average defaults and run the tool.
   - Churn defaults: CreditScore=650, Age=39, Tenure=5, Balance=76485, NumOfProducts=1, HasCrCard=1, IsActiveMember=1, EstimatedSalary=100090, Geography_Germany=0, Geography_Spain=0, Gender_Male=1
   - Diabetes defaults: Pregnancies=3, Glucose=117, BloodPressure=72, SkinThickness=23, Insulin=30.5, BMI=32, Pedigree=0.3725, Age=29
   
3. GEOGRAPHY & GENDER ENCODING: "Germany"->Geography_Germany=1. "Spain"->Geography_Spain=1. "Male"->Gender_Male=1. "Female"->Gender_Male=0.

4. RESPONSE FORMAT - STRICTLY THREE PARAGRAPHS:
   You must format your final response as exactly three distinct paragraphs of plain text separated by a single blank line.
   DO NOT use any labels (like "Paragraph 1:", "Model:").
   DO NOT use any bold text, asterisks, or bullet points.
   
   Paragraph 1: State the prediction result (e.g., SPAM, CHURN) and its confidence level/probability. Explain what this result means in simple terms.
   
   Paragraph 2: State which specific model (e.g., SVM, Neural Network) was used for this prediction and explain why it was chosen based on the user's query.
   
   Paragraph 3: Provide a detailed analysis explaining why the model made this decision based on the specific inputs provided.

5. TOOL USAGE: You must call the appropriate tool FIRST. Only after receiving the tool's output should you generate your 3-paragraph response.

EXAMPLE OF CORRECT FINAL RESPONSE:
Based on the analysis, the customer has a 75.4% probability of churning. This means there is a high likelihood that the customer will close their bank account and leave the bank in the near future.

The prediction was made using the Neural Network (MLP) for Bank Customer Churn model. This model was selected because the query involved banking details such as credit score, balance, and customer retention.

The model made this decision primarily due to the customer's high age and relatively low balance. Even though the customer has a credit card, the combination of a low credit score and short tenure significantly increased the predicted risk of churn."""

# Create the agent using LangChain's new create_agent API
agent_graph = create_agent(llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


class AgentWrapper:
    """Wrapper to make the new LangGraph agent compatible with the app.py interface."""
    
    def invoke(self, inputs):
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        
        # Build messages from chat history
        messages = []
        for msg in chat_history:
            if hasattr(msg, 'type'):
                if msg.type == 'human':
                    messages.append({"role": "user", "content": msg.content})
                elif msg.type == 'ai':
                    messages.append({"role": "assistant", "content": msg.content})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Invoke the agent graph
        result = agent_graph.invoke({"messages": messages})
        
        # Extract the final AI response
        final_messages = result.get("messages", [])
        
        # Get the last AI message
        output = ""
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai' and msg.content:
                output = msg.content
                break
        
        if not output:
            output = "I couldn't generate a response. Please try rephrasing your question."
        
        return {"output": output}


# Export the executor
executor = AgentWrapper()