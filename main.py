from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name = "SmartStudentAgent",
    instructions = """
        You are a smart student assistant.
        you help students by answering acedemic
questions,
        giving effective study tips, and 
        summarizing small text passages clearly.
        """
)

response = Runner.run_sync(
    agent,
    input = "What is Newtom's second law of Motion?",
    run_config = config
)
print(response.final_output)

# Example 2: Study tip
response2 = Runner.run_sync(
    agent,
    input = "Give me a study tip for learning chemistry.",
    run_config = config
)
print(response2.final_output)

# Example 3: Summarization
response3 = Runner.run_sync(
    agent,
    input = "Summarize this: Photosynthesis is the process by which plants use sunlight to produce food from carbon dioxide and water.",
    run_config = config
)
print(response3.final_output)


