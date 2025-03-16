# Game Review Analyzer

A LangChain-based tool for analyzing game reviews from websites, extracting structured data, and generating comprehensive reports.

## Features

- **Website Content Extraction**: Fetches and processes content from game review websites
- **Structured Analysis**: Analyzes review content and extracts structured data in JSON format
- **Report Generation**: Creates detailed reports based on the analysis
- **Multiple LLM Support**: Works with various LLM providers (OpenAI, Google, Groq, OpenRouter, Azure)

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

## License

This project is licensed under the MIT License.