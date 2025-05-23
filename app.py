import streamlit as st
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

# Set up model + search
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
tavily = TavilyClient(api_key=tavily_api_key)

# Research function
def get_research(topic):
    result = tavily.search(query=topic, max_results=5)
    return "\n".join([r["content"] for r in result["results"]])

# Prompt for blog
blog_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template="""
Use this research to write a blog post on: "{topic}"

Research:
{research}

Include:
- Strong headline
- Intro
- 3-5 sections
- Practical advice
- Call-to-action
"""
)

blog_chain = LLMChain(llm=llm, prompt=blog_prompt)

# Repurposing chains
def blog_to_tweets(blog):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="Turn this into a 5-8 tweet thread:\n\n{blog}"
    )
    return LLMChain(llm=llm, prompt=prompt).invoke({"blog": blog})["text"]

def blog_to_caption(blog):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="Write a motivational Instagram/LinkedIn caption from this:\n\n{blog}"
    )
    return LLMChain(llm=llm, prompt=prompt).invoke({"blog": blog})["text"]

def blog_to_script(blog):
    prompt = PromptTemplate(
        input_variables=["blog"],
        template="Write a short (TikTok/Shorts) video script from this:\n\n{blog}"
    )
    return LLMChain(llm=llm, prompt=prompt).invoke({"blog": blog})["text"]

# Streamlit UI
st.set_page_config(page_title="AI Blog Agent", layout="centered")
st.title("🧠 AI Blog + Content Generator")
topic = st.text_input("Enter a blog topic:")

if st.button("Generate"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("🔍 Researching and generating..."):
            research = get_research(topic)
            blog = blog_chain.invoke({"topic": topic, "research": research})["text"]
            tweets = blog_to_tweets(blog)
            caption = blog_to_caption(blog)
            script = blog_to_script(blog)

        st.subheader("📝 Blog Post")
        st.markdown(blog)

        st.subheader("🧵 Twitter Thread")
        st.markdown(tweets)

        st.subheader("📸 Instagram/LinkedIn Caption")
        st.markdown(caption)

        st.subheader("🎬 TikTok/Shorts Script")
        st.markdown(script)
