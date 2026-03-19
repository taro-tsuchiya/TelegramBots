#%%
# used Claude, and verify all lines and largely edited
import re
from typing import List 

def contains_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Detect AI-related keywords in text
    
    Args:
        text: Input text to search
        keywords: List of keywords to search for
    
    Returns:
        matched keywords as a list, or empty set if none found
    """

    
    # convert text to lowercase for matching
    text_lower = text.lower()
    found_keywords = []
    
    # check keywords
    for keyword in keywords:
        escaped_keyword = re.escape(keyword)
        # Allow for slight variations in hyphenation
        # Replace the hyphen or dot with [-\s] --> capture hyphen, white space or no space
        flexible_pattern = escaped_keyword.replace(r'\-', r'[-\s]?')
        # Create word boundary to avoid partial matches (could match e.g., '.')
        pattern = r'\b' + flexible_pattern + r'\b'
        if re.search(pattern, text_lower):
            found_keywords.append(keyword)
    
    return found_keywords


# Example usage and test cases
if __name__ == "__main__":
    # generated test cases by Claude
    test_cases = [
        "I love fresh air in the morning",  # Should be False
        "AIpowered tool is transforming the industry",  # Should be True
        "This is powered by artificial intelligence",  # Should be True
        "Machine learning models are complex",  # Should be True
        "Fair weather today",  # Should be False
        "We use Claude for assistance",  # Should be True
        "The airplane flies high",  # Should be False
        "DALL E generates amazing images",  # Should be True
        "Natural language processing is fascinating",  # Should be True
        "I repaired my car",  # Should be False
        "Deep learning neural networks",  # Should be True
        "OCR technology reads text from images",  # Should be True
    ]
    
    print("Testing AI keyword detection:")
    print("-" * 50)

    ai_keywords = [
    'ai', 'artificial-intelligence', 'machine-learning', 'ml',
    'neural-network', 'deep-learning', 'llm', 'large-language-model',
    'gpt', 'chatgpt', 'claude', 'bard', 'gemini',
    'ocr', 'image-generation', 'text-generation', 'translation',
    'midjourney', 'stable-diffusion', 'deepfake', 'deepfakes',
    'bert', 'roberta', 'embeddings', # 'transformer'
    'nlp', 'natural-language-processing', 'computer-vision',
    'reinforcement-learning', 'supervised-learning', 'unsupervised-learning',
    'generative-ai', 'artificial-neural-network', 'convolutional-neural-network',
    'recurrent-neural-network', 'rnn', 'lstm', 'gru', 'autoencoder',
    'dall-e', 'gpt-3', 'gpt-4',
    'ai-powered', 'ai-generated', 'ai-driven'
    ]

    # numbering starts from 1  
    for i, test_text in enumerate(test_cases, 1):
        matches = contains_keywords(test_text, ai_keywords)
        if len(matches)>0:
            match_bool = True
        else:
            match_bool = False
        print(f"{i:2d}. {'✓' if match_bool else '✗'} '{test_text}'")
        if matches:
            print(f"Found: {', '.join(sorted(matches))}")
# %%
