import sys
import os

# Add current directory to path so we can import backend modules
sys.path.append(os.getcwd())

from backend.openrouter import extract_reasoning

def test_capability_check():
    print("Test 1: Capability Check (Non-reasoning model)")
    content = "Here is the answer. <think>Hidden reasoning</think>"
    message = {}
    model = "openai/gpt-4o" # Not in registry
    
    clean_content, reasoning = extract_reasoning(content, message, model)
    
    if clean_content == content and reasoning == "":
        print("✅ Passed: Ignored tags for non-reasoning model")
    else:
        print(f"❌ Failed: Extracted '{reasoning}' from '{clean_content}'")

def test_field_extraction():
    print("\nTest 2: Field Extraction (Gemini)")
    content = "Final answer"
    message = {"reasoning_details": "This is the reasoning field"}
    model = "google/gemini-3-pro-preview" # In registry with use_field=True
    
    clean_content, reasoning = extract_reasoning(content, message, model)
    
    if reasoning == "This is the reasoning field":
        print("✅ Passed: Extracted from field")
    else:
        print(f"❌ Failed: Got '{reasoning}'")

def test_tag_parsing():
    print("\nTest 3: Tag Parsing (DeepSeek)")
    content = "<think>Step 1: Think\nStep 2: Solve</think>Final Answer"
    message = {}
    model = "deepseek/deepseek-r1" # In registry with parse_tags=True
    
    clean_content, reasoning = extract_reasoning(content, message, model)
    
    if clean_content == "Final Answer" and "Step 1: Think" in reasoning:
        print("✅ Passed: Parsed <think> tags")
    else:
        print(f"❌ Failed: Content='{clean_content}', Reasoning='{reasoning}'")

def test_malformed_tags():
    print("\nTest 4: Malformed Tags")
    content = "<think>Unclosed thinking block... Final Answer"
    message = {}
    model = "deepseek/deepseek-r1"
    
    clean_content, reasoning = extract_reasoning(content, message, model)
    
    # Regex shouldn't match unclosed tag, so content remains same, reasoning empty
    if clean_content == content and reasoning == "":
        print("✅ Passed: Handled unclosed tag gracefully (ignored)")
    else:
        print(f"❌ Failed: Content='{clean_content}', Reasoning='{reasoning}'")

def test_truncation():
    print("\nTest 5: Truncation")
    content = "<think>" + "A" * 3000 + "</think>Answer"
    message = {}
    model = "deepseek/deepseek-r1"
    
    clean_content, reasoning = extract_reasoning(content, message, model)
    
    if len(reasoning) > 2000 and "truncated" in reasoning:
        print("✅ Passed: Truncated long reasoning")
    else:
        print(f"❌ Failed: Length {len(reasoning)}")

if __name__ == "__main__":
    try:
        test_capability_check()
        test_field_extraction()
        test_tag_parsing()
        test_malformed_tags()
        test_truncation()
    except Exception as e:
        print(f"❌ Error: {e}")
