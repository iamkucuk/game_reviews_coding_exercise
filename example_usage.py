#!/usr/bin/env python3
from game_review_analyzer import process_game_reviews

def main():
    # Example URLs to process
    urls = [
        "https://www.gamersheroes.com/honest-game-reviews/babylons-fall-review/",
        "https://www.digitallydownloaded.net/2022/03/review-babylons-fall-sony-playstation-5.html",
        "https://cogconnected.com/review/babylons-fall-review-an-encumbered-experience/",
        "https://themakoreactor.com/featured/babylons-fall-review-ps5-dualsense-nier-automata-postgame-story-crafting-ps4-steam-crossplay/35999/",
        "https://www.destructoid.com/reviews/review-babylons-fall/",
        "https://www.gamespew.com/2022/03/babylons-fall-review/",
        "https://www.shacknews.com/article/129272/babylons-fall-review-an-unworthy-endeavor",
        "https://www.playstationlifestyle.net/review/863086-babylons-fall-review-ps5-platinum-games-fall-from-grace/#/slide/1",
        "https://twinfinite.net/reviews/babylons-fall-review-crash-and-burn/",
        "https://attackofthefanboy.com/reviews/babylons-fall-review/",
        "https://www.ign.com/articles/babylons-fall-review",
        "https://www.psu.com/reviews/babylons-fall-review-ps5/",
        "https://ztgd.com/reviews/babylons-fall-ps5/",
        "https://www.gameskinny.com/reviews/babylons-fall-review-falling-short/",
        "https://checkpointgaming.net/reviews/2022/03/babylons-fall-review-how-the-mighty-have-fallen/",
        "https://worthplaying.com/article/2022/4/1/reviews/131358-ps5-review-babylons-fall/",
        "https://www.godisageek.com/reviews/babylons-fall-review/",
        "https://www.thesixthaxis.com/2022/03/07/babylons-fall-review/",
        "https://www.well-played.com.au/babylons-fall-review/",
        "https://www.pushsquare.com/reviews/ps5/babylons-fall",
        "https://www.videogameschronicle.com/review/babylons-fall/",
        "https://www.dexerto.com/reviews/babylons-fall-review-1774470/",
        "https://www.cgmagonline.com/review/game/babylons-fall-ps5-review/",
        "https://metro.co.uk/2022/03/09/babylons-fall-review-the-descent-of-platinum-games-16244599/",
        "https://screenrant.com/babylons-fall-game-review/",
        "https://themakoreactor.com/featured/babylons-fall-review-ps5-dualsense-nier-automata-postgame-story-crafting-ps4-steam-crossplay/35999/",
        "https://www.destructoid.com/reviews/review-babylons-fall/",
        "https://www.gamespew.com/2022/03/babylons-fall-review/",
    ]
    
    # Analysis prompt that will be used to extract structured data from the review
    with open("analysis_prompt.txt", "r") as file:
        analysis_prompt = file.read()
    
    # Report prompt that will generate the final analysis report
    with open("report_prompt.txt", "r") as file:
        report_prompt = file.read()
    
    # Choose your preferred LLM provider:
    # Options: "openai", "google", "groq", "openrouter", or "azure"
    llm_provider = "groq"
    
    # Specify a model when using OpenRouter
    # Examples:
    # - "anthropic/claude-3.5-sonnet"
    # - "openai/gpt-4o"
    # - "meta-llama/llama-3-70b-instruct"
    # - "anthropic/claude-3-opus"
    # model_name = "anthropic/claude-3.5-sonnet"
    
    # Process the reviews
    results = process_game_reviews(urls, analysis_prompt, report_prompt, llm_provider)
    
    # Print results summary
    print("\n===== Processing Complete =====")
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Analysis file: {result['analysis_json_file']}")
        print(f"Report file: {result['report_file']}")
        print("-----------------------------")

if __name__ == "__main__":
    main()