from crewai import Agent
from tools.search_tools import SearchTools
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
gemini_api_key = os.environ.get('GEMINI_API_KEY')

class AINewsLetterAgents():
    def __init__(self):
        #self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        #self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        #self.Ollama = Ollama(model="openhermes")
        self.llm_gemini = ChatGoogleGenerativeAI(
            model= "gemini-pro", verbose = True, temperatur = 0.1, google_api_key= gemini_api_key
        )

    def editor_agent(self):
        return Agent(
            role='Editor',
            goal='Oversee the creation of the AI Newsletter',
            backstory="""With a keen eye for detail and a passion for storytelling, you ensure that the newsletter
            not only informs but also engages and inspires the readers. """,
            allow_delegation=True,
            verbose=True,
            max_iter=15,
            llm=self.llm_gemini, #2024029
        )

    def news_fetcher_agent(self):
        return Agent(
            role='NewsFetcher',
            goal='Fetch the top AI news stories for the day',
            backstory="""As a digital sleuth, you scour the internet for the latest and most impactful developments
            in the world of AI, ensuring that our readers are always in the know.""",
            tools=[SearchTools.search_internet],
            verbose=True,
            allow_delegation=True,
            llm=self.llm_gemini, #2024029
        )

    def news_analyzer_agent(self):
        return Agent(
            role='NewsAnalyzer',
            goal='Analyze each news story and generate a detailed markdown summary',
            backstory="""With a critical eye and a knack for distilling complex information, you provide insightful
            analyses of AI news stories, making them accessible and engaging for our audience.""",
            tools=[SearchTools.search_internet],
            verbose=True,
            allow_delegation=True,
            llm=self.llm_gemini, #2024029
        )

    def newsletter_compiler_agent(self):
        return Agent(
            role='NewsletterCompiler',
            goal='Compile the analyzed news stories into a final newsletter format',
            backstory="""As the final architect of the newsletter, you meticulously arrange and format the content,
            ensuring a coherent and visually appealing presentation that captivates our readers. Make sure to follow
            newsletter format guidelines and maintain consistency throughout.""",
            verbose=True,
            llm=self.llm_gemini, #2024029
        )
