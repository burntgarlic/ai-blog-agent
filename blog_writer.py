from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from tavily import TavilyClient

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Set up the model
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)

# Tavily client for web research
tavily = TavilyClient(api_key=tavily_api_key)

# Function to search
def get_research(topic):
    result = tavily.search(query=topic, max_results=5)
    return "\n".join([r["content"] for r in result["results"]])

# Prompt template
prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template="""
You are an expert blog writer.

Use the following research to inform your writing:
{research}

Now write a helpful blog post on: "{topic}"

Include:
- A strong headline
- A clear introduction
- 3-5 key sections with subheadings
- Practical advice
- A call-to-action at the end
"""
)

# Wrap in chain
chain = LLMChain(llm=llm, prompt=prompt)

def blog_to_tweets(blog_text):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="""
You are a Twitter expert.

Turn this blog into a 5-8 tweet thread. Each tweet should be helpful, engaging, and standalone if possible. Start with a strong hook.

BLOG:
{blog}
"""
    )
    tweet_chain = LLMChain(llm=llm, prompt=prompt)
    result = tweet_chain.invoke({"blog": blog_text})
    return result["text"]

def blog_to_caption(blog_text):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="""
Turn the blog below into a motivational, friendly Instagram or LinkedIn caption.

Make it:
- Conversational
- Helpful
- Uplifting
- End with a relatable call-to-action

BLOG:
{blog}
"""
    )
    caption_chain = LLMChain(llm=llm, prompt=prompt)
    result = caption_chain.invoke({"blog": blog_text})
    return result["text"]

def blog_to_script(blog_text):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="""
Turn the blog below into a short video script (for TikTok or YouTube Shorts).

Format:
- Hook
- Quick facts or value
- Call-to-action

Use punchy, informal language. Script should be 45-60 seconds max.

BLOG:
{blog}
"""
    )
    script_chain = LLMChain(llm=llm, prompt=prompt)
    result = script_chain.invoke({"blog": blog_text})
    return result["text"]


# Main execution
if __name__ == "__main__":
    topic = input("Enter a blog topic: ")
    research = get_research(topic)
    blog = chain.invoke({"topic": topic, "research": research})

    print("\n===== YOUR BLOG POST =====\n")
    print(blog["text"])

    print("\n===== TWEET THREAD =====\n")
    print(blog_to_tweets(blog["text"]))

    print("\n===== INSTAGRAM/LINKEDIN CAPTION =====\n")
    print(blog_to_caption(blog["text"]))

    print("\n===== VIDEO SCRIPT (SHORTS/TIKTOK) =====\n")
    print(blog_to_script(blog["text"]))
