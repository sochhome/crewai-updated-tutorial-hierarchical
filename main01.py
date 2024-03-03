# 20240229
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks
from file_io import save_markdown

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
gemini_api_key = os.environ.get('GEMINI_API_KEY')

# Initialize the agents and tasks
agents = AINewsLetterAgents()
tasks = AINewsLetterTasks()

# # Initialize the OpenAI GPT-4 language model
# OpenAIGPT4 = ChatOpenAI(
#     model="gpt-4"
# )

# 20240229
llm_gemini = ChatGoogleGenerativeAI(
    model= "gemini-pro", verbose = True, temperatur = 0.1, google_api_key= gemini_api_key
)


# Instantiate the agents
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_compiler = agents.newsletter_compiler_agent()

# Instantiate the tasks
fetch_news_task = tasks.fetch_news_task(news_fetcher)
analyze_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task]) # [fetch_news_task]-->context
compile_newsletter_task = tasks.compile_newsletter_task(
    newsletter_compiler, [analyze_news_task], save_markdown)

# Form the crew
crew = Crew(
    agents=[editor, news_fetcher, news_analyzer, newsletter_compiler],
    tasks=[fetch_news_task, analyze_news_task, compile_newsletter_task],
    process=Process.hierarchical,
    manager_llm= llm_gemini, # OpenAIGPT4,
    verbose=2
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:")
print(results)
