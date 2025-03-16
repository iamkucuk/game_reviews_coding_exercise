# Game Review Analyzer

A LangChain-based tool for analyzing game reviews from websites, extracting structured data, and generating comprehensive reports.

## Features

- **Website Content Extraction**: Fetches and processes content from game review websites using `requests` and `Beautiful Soup`.
- **Structured Analysis**: Analyzes review content and extracts structured data in JSON format, categorizing text segments by sentiment, category, and sub-category.
- **Report Generation**: Creates detailed reports based on the analysis, summarizing key positive and negative aspects of the game.
- **Multiple LLM Support**: Works with various LLM providers, including OpenAI, Google Gemini, Groq, OpenRouter, and Azure OpenAI.
- **Rate Limiting**: Implements rate limiting using the `RateLimiter` class to manage API call frequency.
- **JSON Output**: Ensures the output is valid JSON with specific formatting instructions and retry mechanisms.
- **Review Enhancement**: Includes a review enhancement step to identify and incorporate missing review segments.

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and add your API keys:
   ```
   cp .env.example .env
   ```

## Usage

### Basic Usage

The script can analyze multiple game review websites sequentially:

```python
from game_review_analyzer import process_game_reviews

# List of URLs to process
urls = [
    "https://www.ign.com/articles/the-legend-of-zelda-tears-of-the-kingdom-review",
]

# Prompt for the analysis phase
analysis_prompt = """
Extract all segments from the game review that contain opinions, facts, or assessments about the game.
For each segment:
- 'text' should be the exact quote from the review
- 'category' should be one of: Gameplay, Graphics, Story, Audio, Performance, Value
- 'sentiment' should be: positive, neutral, or negative
- 'sub_category' should be a more specific aspect of the category
"""

# Prompt for the report generation phase
report_prompt = """
Create a comprehensive summary of the game review, highlighting the main positive and negative aspects.
Include separate sections for each main category (Gameplay, Graphics, Story, etc.).
End with an overall assessment of the game based on the review.
"""

# Choose LLM provider (openai, google, groq, openrouter, azure)
provider = "openai"

# Process the reviews
results = process_game_reviews(urls, analysis_prompt, report_prompt, provider)
```

### Example Script

The repository includes an example script (`example_usage.py`) that demonstrates how to use the analyzer:

```
python example_usage.py
```

## Output Format

### Analysis JSON

The analysis is saved in a structured JSON format:

```json
{
  "reviews": [
    {
      "text": "exact text part used for coding",
      "category": "one of main categories",
      "sentiment": "positive|neutral|negative",
      "sub_category": "one of the sub categories of the corresponding category"
    },
    ...
  ]
}
```

### Report

The generated report is saved as a text file with a comprehensive analysis of the game review.

## Supported LLM Providers

- OpenAI (default)
- Google AI (Gemini)
- Groq
- OpenRouter
- Azure OpenAI

## Rate Limiting

The tool implements rate limiting using the `RateLimiter` class from rate_limiter.py to manage API call frequency. The default rate limit is set to 100 requests per minute.

## JSON Output and Retries

The tool is designed to ensure that the LLM returns valid JSON output. It includes specific formatting instructions in the prompt and implements retry mechanisms to handle cases where the LLM returns invalid JSON or data with an incorrect structure. The `execute_llm_for_json` function in game_review_analyzer.py handles this process.

## Review Enhancement

The tool includes a review enhancement step to identify and incorporate missing review segments. After the initial analysis, the LLM re-examines the original game review content to find any important segments that were missed. The `merge_reviews` function in game_review_analyzer.py merges the initial and enhanced reviews, avoiding duplicates.

## License

This project is licensed under the MIT License.
