from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
import os

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define a heavy workload tool
def summarize_email(email: str):
    return f"Summary of: {email[:60]}..."

# Register tool
tools = [
    Tool(
        name="summarize_email",
        func=summarize_email,
        description="Summarizes a single email message"
    )
]

# Initialize agent
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# User task that results in excessive tool calls
prompt = '''
You are managing a shared inbox of 100 emails. For each email, please generate 5 different summaries to explore tone and clarity.

Start immediately.
'''

agent.run(prompt)