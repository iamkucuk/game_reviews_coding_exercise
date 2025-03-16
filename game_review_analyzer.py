#!/usr/bin/env python3
import json
import os
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import AzureChatOpenAI
from os import getenv, makedirs
from dotenv import load_dotenv
from rate_limiter import RateLimiter

# Load environment variables from .env file
load_dotenv()

# Create rate limiter for LLM calls (100 requests per minute)
llm_rate_limiter = RateLimiter(max_calls=100, period=60)

# Define the Pydantic model for our game review analysis
class ReviewSegment(BaseModel):
    text: str = Field(description="Exact text part used for coding")
    category: str = Field(description="One of main categories")
    sentiment: str = Field(description="Sentiment: positive, neutral, or negative")
    sub_category: str = Field(description="One of the sub categories of the corresponding category")

class ReviewAnalysis(BaseModel):
    reviews: List[ReviewSegment] = Field(description="List of analyzed review segments")

# Global constants for max tokens
MAX_OUTPUT_TOKENS = {
    "openai": 8192,         # GPT-4 max tokens
    "google": 38192,         # Gemini max tokens
    "groq": 38192,           # Groq max tokens
    "openrouter": 8192,     # Default for OpenRouter
    "azure": 8192,          # Azure OpenAI default
}

# Ensure output directories exist
def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs("content", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    os.makedirs("report", exist_ok=True)

# Get website name from URL for file naming
def get_website_name(url: str) -> str:
    """Extract a clean website name from URL for file naming"""
    # Remove protocol
    name = url.replace("http://", "").replace("https://", "")
    # Remove www if present
    name = name.replace("www.", "")
    # Get domain part
    parts = name.split("/")
    if len(parts) > 0:
        domain = parts[0]
        # Extract the main domain name without TLD
        domain_parts = domain.split(".")
        if len(domain_parts) > 1:
            return domain_parts[-2]  # Return the main domain name
    # Fallback to a sanitized version of the full URL
    return url.replace("://", "_").replace("/", "_").replace(".", "_")

# Function to fetch website content
def fetch_website_content(url: str, max_retries: int = 3) -> str:
    """
    Fetch the content of a website given its URL with retry logic.
    
    Args:
        url: URL of the website to fetch
        max_retries: Maximum number of retry attempts
        
    Returns:
        Extracted text content from the website or empty string if failed
    """
    retry_count = 0
    backoff_factor = 2
    initial_delay = 1
    
    while retry_count < max_retries:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup to extract text content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text (remove extra whitespace)
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                delay = initial_delay * (backoff_factor ** (retry_count - 1))
                print(f"Error fetching website content: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"Failed to fetch website content after {max_retries} attempts: {e}")
                return ""

# Function to get OpenRouter instance
def get_openrouter(model: str = "deepseek/deepseek-r1:free") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=MAX_OUTPUT_TOKENS["openrouter"]
    )

# Function to select an LLM based on provider
def get_llm(provider: str = "openai", model: str = None):
    """Get an LLM instance based on the specified provider."""
    if provider == "openai":
        return ChatOpenAI(temperature=0, max_tokens=MAX_OUTPUT_TOKENS["openai"])
    elif provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            openai_api_version=os.environ.get("OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS["azure"]
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS["google"])
    elif provider == "groq":
        return ChatGroq(model_name="qwen-qwq-32b", temperature=0, max_tokens=MAX_OUTPUT_TOKENS["groq"])
    elif provider == "openrouter":
        if model:
            return get_openrouter(model=model)
        else:
            return get_openrouter(model="deepseek/deepseek-r1:free")
    else:
        # Default to OpenAI if provider not recognized
        print(f"Provider {provider} not recognized, using OpenAI.")
        return ChatOpenAI(temperature=0, max_tokens=MAX_OUTPUT_TOKENS["openai"])

# Execute LLM call with rate limiting and retry logic
@llm_rate_limiter
def execute_llm_with_retry(chain, inputs, max_retries: int = 3):
    """
    Execute an LLM chain with retry logic and rate limiting.
    
    Args:
        chain: The LangChain chain to execute
        inputs: The inputs to the chain
        max_retries: Maximum number of retry attempts
        
    Returns:
        The result of the chain execution or None if failed
    """
    retry_count = 0
    backoff_factor = 2
    initial_delay = 1
    
    while retry_count < max_retries:
        try:
            return chain.invoke(inputs)
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                delay = initial_delay * (backoff_factor ** (retry_count - 1))
                print(f"LLM request failed: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"LLM request failed after {max_retries} attempts: {e}")
                return None

# Execute LLM call until valid JSON is returned
@llm_rate_limiter
def execute_llm_for_json(llm, prompt_template, inputs, max_retries: int = 5):
    """
    Execute an LLM chain until it returns valid JSON with the expected structure.
    
    Args:
        llm: The language model to use
        prompt_template: The prompt template to use
        inputs: The inputs to the template
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple containing (raw_response, json_data) or (error_message, None) if failed
    """
    retry_count = 0
    backoff_factor = 1.5
    initial_delay = 1
    
    # Create enhanced prompt with explicit JSON formatting instructions
    enhanced_prompt = PromptTemplate(
        template=prompt_template.template + "\n\nYour response MUST be valid JSON that can be parsed directly. Format your response as a JSON object with a 'reviews' array.",
        input_variables=prompt_template.input_variables
    )
    
    # Create the chain
    chain = enhanced_prompt | llm
    
    while retry_count < max_retries:
        try:
            # Get response from LLM
            raw_response = chain.invoke(inputs)
            raw_response = raw_response.content
            raw_response = raw_response.replace('```json', '')  # Remove code block formatting
            raw_response = raw_response.replace('```', '')  # Remove code block formatting
            raw_response = raw_response.split("</think>", 1)[1]
            # Try to parse JSON directly
            try:
                json_data = json.loads(raw_response)
                
                # Check if the response has the expected structure
                if "reviews" in json_data and isinstance(json_data["reviews"], list):
                    # Valid JSON with expected structure
                    return raw_response, json_data
                else:
                    print(f"LLM returned valid JSON but without the expected 'reviews' array structure. Retrying...")
            except json.JSONDecodeError as e:
                print(f"LLM returned invalid JSON: {e}. Retrying...")
            
            # If we get here, the response wasn't valid JSON or had wrong structure
            retry_count += 1
            if retry_count < max_retries:
                delay = initial_delay * (backoff_factor ** (retry_count - 1))
                print(f"Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"Failed to get valid JSON after {max_retries} attempts")
                return raw_response, None
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                delay = initial_delay * (backoff_factor ** (retry_count - 1))
                print(f"LLM request failed: {e}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"LLM request failed after {max_retries} attempts: {e}")
                return f"Failed to get response from LLM after {max_retries} attempts: {e}", None
    
    return f"Failed to get valid JSON after {max_retries} attempts", None

# Merge reviews from initial analysis and enhancement
def merge_reviews(initial_reviews: List[Dict[str, Any]], enhancement_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge reviews from initial analysis and enhancement, avoiding duplicates.
    
    Args:
        initial_reviews: List of review segments from initial analysis
        enhancement_reviews: List of review segments from enhancement
        
    Returns:
        Combined list of review segments without duplicates
    """
    # Create a set of existing review texts to check for duplicates
    existing_texts = {review["text"] for review in initial_reviews}
    
    # Add unique reviews from enhancement
    merged_reviews = initial_reviews.copy()
    for review in enhancement_reviews:
        if review["text"] not in existing_texts:
            merged_reviews.append(review)
            existing_texts.add(review["text"])
    
    return merged_reviews

# Step 1: Process website and generate JSON analysis
def analyze_game_review(url: str, prompt: str, provider: str = "openai", model: str = None) -> tuple:
    """
    Analyze game review website and generate structured JSON analysis.
    
    Args:
        url: URL of the game review website
        prompt: Prompt to use for the LLM
        provider: LLM provider to use
        model: Specific model to use (for OpenRouter)
        
    Returns:
        Tuple containing (content, raw_response, json_result)
    """
    # Fetch website content
    content = fetch_website_content(url)
    
    if not content:
        return content, "", json.dumps({"error": "Failed to fetch website content"})
    
    # Save content to file
    website_name = get_website_name(url)
    content_file = os.path.join("content", f"{website_name}.txt")
    with open(content_file, "w") as f:
        f.write(content)
    print(f"Content saved to {content_file}")
    
    # Initialize the LLM
    llm = get_llm(provider, model)
    
    # Setup prompt template
    prompt_template = PromptTemplate(
        template="""
        Analyze the following game review content and extract segments that contain meaningful opinions or information.
        For each relevant segment, determine its main category, subcategory, and sentiment.
        
        Website content:
        {content}
        
        User instructions:
        {prompt}
        
        Return the analysis as a JSON object with a 'reviews' array, where each item contains 'text', 'category', 'sentiment', and 'sub_category'.
        """,
        input_variables=["content", "prompt"]
    )
    
    # Execute until we get valid JSON
    print("Getting initial review analysis...")
    raw_response, json_result = execute_llm_for_json(
        llm, 
        prompt_template, 
        {"content": content, "prompt": prompt}
    )
    
    if json_result is None:
        return content, raw_response, json.dumps({"error": "Failed to get valid JSON analysis"})
    
    # Save the initial parsed reviews for the enhancement step
    initial_reviews = json_result.get("reviews", [])
    
    # Step 1.5: Enhancement - Ask LLM to find missing review segments
    print("Step 1.5: Enhancing review analysis - checking for missing segments...")
    
    # Setup enhancement prompt template
    enhancement_prompt_template = PromptTemplate(
        template="""
        You previously analyzed a game review with the following instructions:
        
        {original_prompt}
        
        And you provided this analysis:
        
        {initial_analysis}
        
        Re-examine the original game review content carefully and identify any important segments that were missed in your initial analysis. Focus on:
        
        1. Significant opinions or facts that weren't captured
        2. Important aspects of gameplay, graphics, story, or other categories not mentioned
        3. Nuanced sentiments or subcategories that would enhance the analysis
        
        Original game review content:
        {content}
        
        Return ONLY the missing review segments as a JSON object with a 'reviews' array, following the same format as your initial analysis. 
        Each item should contain 'text', 'category', 'sentiment', and 'sub_category'. If there are no missing segments, return an empty reviews array.
        
        Be thorough and make sure to capture segments that contain different perspectives or aspects not covered in the initial analysis.
        """,
        input_variables=["original_prompt", "initial_analysis", "content"]
    )
    
    # Execute enhancement with retry logic for JSON
    enhancement_raw, enhancement_result = execute_llm_for_json(
        llm,
        enhancement_prompt_template,
        {
            "original_prompt": prompt, 
            "initial_analysis": raw_response,
            "content": content
        }
    )
    
    # Get enhancement reviews or empty list if failed
    enhancement_reviews = []
    if enhancement_result and "reviews" in enhancement_result:
        enhancement_reviews = enhancement_result.get("reviews", [])
    
    # Merge the initial and enhancement reviews
    if enhancement_reviews:
        print(f"Found {len(enhancement_reviews)} additional review segments")
        all_reviews = merge_reviews(initial_reviews, enhancement_reviews)
        
        # Create the final combined JSON result
        final_json_result = {"reviews": all_reviews}
    else:
        print("No additional review segments found")
        final_json_result = {"reviews": initial_reviews}
    
    # Return the combined results
    return content, raw_response, json.dumps(final_json_result, indent=2)

# Step 2: Generate report from analysis
def generate_report(url: str, content: str, analysis_json: str, report_prompt: str, provider: str = "openai", model: str = None) -> str:
    """
    Generate a comprehensive report based on the analysis.
    
    Args:
        url: URL of the game review website
        content: Website content (already fetched)
        analysis_json: JSON string containing the analysis
        report_prompt: Prompt to use for generating the report
        provider: LLM provider to use
        model: Specific model to use (for OpenRouter)
        
    Returns:
        Generated report text
    """
    if not content:
        return "Failed to fetch website content for report generation."
    
    # Initialize the LLM
    llm = get_llm(provider, model)
    
    # Setup prompt template for report generation
    prompt_template = PromptTemplate(
        template="""
        You are tasked with creating a comprehensive report about a game review.
        
        Original website content:
        {content}
        
        Analysis of the review:
        {analysis}
        
        User instructions for report:
        {report_prompt}
        
        Generate a detailed, well-structured report based on the above information.
        """,
        input_variables=["content", "analysis", "report_prompt"]
    )
    
    # Create the chain
    chain = prompt_template | llm
    
    # Execute with retry logic and rate limiting
    report = execute_llm_with_retry(chain, {
        "content": content,  # Send the entire content
        "analysis": analysis_json,
        "report_prompt": report_prompt
    })
    
    if report is None:
        return "Failed to generate report after multiple attempts."
    
    return report

# Main function to process a list of game review websites
def process_game_reviews(urls: List[str], analysis_prompt: str, report_prompt: str, provider: str = "openai", model: str = None):
    """
    Process multiple game review websites.
    
    Args:
        urls: List of URLs to process
        analysis_prompt: Prompt for the analysis stage
        report_prompt: Prompt for the report generation stage
        provider: LLM provider to use
        model: Specific model to use (for OpenRouter)
    """
    # Ensure output directories exist
    ensure_output_dirs()
    
    results = []
    
    for url in urls:
        print(f"\nProcessing: {url}")
        website_name = get_website_name(url)
        
        # Step 1: Analyze the game review (includes enhancement step 1.5)
        print("Step 1: Analyzing game review...")
        content, raw_analysis, analysis_json = analyze_game_review(url, analysis_prompt, provider, model)
        
        # Save raw analysis text
        analysis_text_file = os.path.join("analysis", f"{website_name}.txt")
        with open(analysis_text_file, "w") as f:
            f.write(raw_analysis)
        print(f"Raw analysis saved to {analysis_text_file}")
        
        # Save parsed JSON
        analysis_json_file = os.path.join("analysis", f"{website_name}.json")
        with open(analysis_json_file, "w") as f:
            f.write(analysis_json)
        print(f"Analysis JSON saved to {analysis_json_file}")
        
        # Step 2: Generate report
        print("Step 2: Generating report...")
        report = generate_report(url, content, analysis_json, report_prompt, provider, model)
        report = getattr(report, 'content', report)
        report = report.split("</think>", 1)[-1]
        report = report.replace('```markdown', '')
        report = report.replace('```', '')
        report = report.strip()

        # Save report
        report_file = os.path.join("report", f"{website_name}.md")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to {report_file}")
        
        results.append({
            "url": url,
            "content_file": os.path.join("content", f"{website_name}.txt"),
            "analysis_text_file": analysis_text_file,
            "analysis_json_file": analysis_json_file,
            "report_file": report_file
        })
    
    return results

if __name__ == "__main__":
    # Example usage
    urls = [
        # Add your game review URLs here
        # For example: "https://www.ign.com/games/reviews/zelda-tears-of-the-kingdom"
    ]
    
    analysis_prompt = """
    Extract all segments from the game review that contain opinions, facts, or assessments about the game.
    For each segment:
    - 'text' should be the exact quote from the review
    - 'category' should be one of: Gameplay, Graphics, Story, Audio, Performance, Value
    - 'sentiment' should be: positive, neutral, or negative
    - 'sub_category' should be a more specific aspect of the category (e.g., for Gameplay: Controls, Level Design, etc.)
    """
    
    report_prompt = """
    Create a comprehensive summary of the game review, highlighting the main positive and negative aspects.
    Include separate sections for each main category (Gameplay, Graphics, Story, etc.).
    End with an overall assessment of the game based on the review.
    """
    
    # Specify the LLM provider: "openai", "google", "groq", "openrouter", or "azure"
    llm_provider = "openai"
    # Optional: specify a model when using OpenRouter
    model_name = None  # e.g., "anthropic/claude-3.5-sonnet"
    
    # Process the reviews
    process_game_reviews(urls, analysis_prompt, report_prompt, llm_provider, model_name)