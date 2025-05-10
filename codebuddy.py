import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Provider
provider = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = provider
)

# Config : Define at run level
run_config = RunConfig(
    model = model,
    model_provider = provider,
    tracing_disabled = True
)

# Choose a Domain
# Let's go with Education — for example, an agent that helps students understand Python concepts. 

codebuddy = Agent(
    name = "CodeBuddy",
    instructions = """
    You are codebuddy, a friendly and knowledgeable python programming assistant.
    Your job is help to beginners understand python concepts, syntax and provide simple example code.
    Avoid technical jargon and always explain things in a simple and clear way.
    Never overwhelm the user provide short, digestible answers.
    And each response with a related python tip or encouragement"""
)

# Example prompt

user_prompt = "can you explain what a python list is?"
# user_prompt = "explain strings in python"

result = Runner.run_sync(
    codebuddy,
     user_prompt,
    run_config = run_config

)

# save output to output.md
with open("output.md", "a", encoding="utf-8") as f:
    f.write(result.final_output)

# Also print the result
print(result.final_output)

