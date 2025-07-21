"""
Data utilities for the MARU project.

This module contains functions for generating synthetic datasets, particularly
the "Needle-in-Haystack" task for evaluating sequence models.
"""

import random
import torch
from typing import List, Tuple, Union, Optional, Dict, Any
import string
import re
import json
import time
import ollama
from datetime import datetime


def generate_needle_haystack_task(
    haystack_len: int,
    needle_string: str,
    vocabulary: Union[List[str], str],
    random_seed: Optional[int] = None
) -> Tuple[str, str, int]:
    """
    Generate a "Needle-in-Haystack" task instance.
    
    This function creates a long random sequence (the 'haystack') and inserts
    a specific string (the 'needle') at a random position within it.
    
    Args:
        haystack_len: Total length of the haystack sequence
        needle_string: The needle string to insert
        vocabulary: List or string of characters to use for generating the haystack
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple containing:
        - full_sequence: The complete sequence with needle inserted
        - needle: The needle string (same as input)
        - start_index: Starting position of the needle in the sequence
        
    Raises:
        ValueError: If needle is longer than haystack or other invalid parameters
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Validate inputs
    if len(needle_string) >= haystack_len:
        raise ValueError(f"Needle length ({len(needle_string)}) must be less than haystack length ({haystack_len})")
    
    if not needle_string:
        raise ValueError("Needle string cannot be empty")
    
    if not vocabulary:
        raise ValueError("Vocabulary cannot be empty")
    
    # Convert vocabulary to list if it's a string
    if isinstance(vocabulary, str):
        vocab_chars = list(set(vocabulary))
    else:
        vocab_chars = list(vocabulary)
    
    if not vocab_chars:
        raise ValueError("Vocabulary must contain at least one character")
    
    # Calculate available positions for needle insertion
    max_start_pos = haystack_len - len(needle_string)
    if max_start_pos < 0:
        raise ValueError("Haystack too short to contain needle")
    
    # Choose random position for needle
    needle_start_pos = random.randint(0, max_start_pos)
    
    # Generate random characters for the haystack
    # We'll generate the full sequence and then insert the needle
    full_sequence_chars = []
    
    # Generate characters before needle
    for i in range(needle_start_pos):
        full_sequence_chars.append(random.choice(vocab_chars))
    
    # Insert needle
    full_sequence_chars.extend(list(needle_string))
    
    # Generate characters after needle
    remaining_length = haystack_len - needle_start_pos - len(needle_string)
    for i in range(remaining_length):
        full_sequence_chars.append(random.choice(vocab_chars))
    
    full_sequence = ''.join(full_sequence_chars)
    
    # Verify the result
    assert len(full_sequence) == haystack_len
    assert full_sequence[needle_start_pos:needle_start_pos + len(needle_string)] == needle_string
    
    return full_sequence, needle_string, needle_start_pos


def generate_needle_haystack_batch(
    batch_size: int,
    haystack_len: int,
    needle_string: str,
    vocabulary: Union[List[str], str],
    random_seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[int]]:
    """
    Generate a batch of "Needle-in-Haystack" task instances.
    
    Args:
        batch_size: Number of instances to generate
        haystack_len: Total length of each haystack sequence
        needle_string: The needle string to insert
        vocabulary: List or string of characters to use for generating haystacks
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple containing:
        - sequences: List of full sequences
        - needles: List of needle strings (all same as input)
        - positions: List of starting positions for each needle
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    sequences = []
    needles = []
    positions = []
    
    for i in range(batch_size):
        # Use different seed for each instance if base seed provided
        instance_seed = None if random_seed is None else random_seed + i
        
        sequence, needle, position = generate_needle_haystack_task(
            haystack_len=haystack_len,
            needle_string=needle_string,
            vocabulary=vocabulary,
            random_seed=instance_seed
        )
        
        sequences.append(sequence)
        needles.append(needle)
        positions.append(position)
    
    return sequences, needles, positions


def generate_variable_needle_haystack_batch(
    batch_size: int,
    haystack_len_range: Tuple[int, int],
    needle_strings: List[str],
    vocabulary: Union[List[str], str],
    random_seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Generate a batch with variable haystack lengths and different needles.
    
    Args:
        batch_size: Number of instances to generate
        haystack_len_range: Tuple of (min_length, max_length) for haystacks
        needle_strings: List of possible needle strings to choose from
        vocabulary: List or string of characters to use for generating haystacks
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple containing:
        - sequences: List of full sequences
        - needles: List of needle strings used
        - positions: List of starting positions for each needle
        - lengths: List of haystack lengths used
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    min_len, max_len = haystack_len_range
    
    sequences = []
    needles = []
    positions = []
    lengths = []
    
    for i in range(batch_size):
        # Choose random haystack length and needle
        haystack_len = random.randint(min_len, max_len)
        needle_string = random.choice(needle_strings)
        
        # Skip if needle is too long for this haystack
        if len(needle_string) >= haystack_len:
            # Choose a shorter needle or longer haystack
            valid_needles = [n for n in needle_strings if len(n) < haystack_len]
            if not valid_needles:
                haystack_len = max(haystack_len, len(needle_string) + 10)
            else:
                needle_string = random.choice(valid_needles)
        
        # Use different seed for each instance if base seed provided
        instance_seed = None if random_seed is None else random_seed + i
        
        sequence, needle, position = generate_needle_haystack_task(
            haystack_len=haystack_len,
            needle_string=needle_string,
            vocabulary=vocabulary,
            random_seed=instance_seed
        )
        
        sequences.append(sequence)
        needles.append(needle)
        positions.append(position)
        lengths.append(haystack_len)
    
    return sequences, needles, positions, lengths


def create_default_vocabulary() -> List[str]:
    """
    Create a default vocabulary for needle-in-haystack tasks.
    
    Returns:
        List of characters including letters, digits, and some punctuation
    """
    # Use printable ASCII characters excluding some special ones
    chars = string.ascii_letters + string.digits + ".,!?;:-_"
    return list(set(chars))


def create_simple_vocabulary() -> List[str]:
    """
    Create a simple vocabulary with just letters and digits.

    Returns:
        List of alphanumeric characters
    """
    return list(string.ascii_letters + string.digits)


def format_needle_haystack_prompt(haystack: str, needle: str) -> str:
    """
    Format a (haystack, needle) pair into a clear prompt for the teacher model.

    Args:
        haystack: The full sequence containing the needle
        needle: The needle string to find

    Returns:
        Formatted prompt string for the teacher model
    """
    prompt = f"""You are helping with a sequence analysis task. I will give you a long text sequence and ask you to find the starting position of a specific substring within it.

TEXT SEQUENCE:
{haystack}

FIND THIS SUBSTRING:
{needle}

Please find the starting position (0-indexed) of the substring "{needle}" in the text sequence above.

Your response should include:
1. The starting position as a number
2. A brief explanation of your reasoning

IMPORTANT: The position should be 0-indexed (first character is position 0).

Starting position:"""

    return prompt


def parse_teacher_response(response_text: str, needle: str, haystack: str) -> Tuple[Optional[int], str]:
    """
    Parse the teacher model's response to extract the numerical answer.

    Args:
        response_text: The full response from the teacher model
        needle: The needle string (for verification)
        haystack: The haystack string (for verification)

    Returns:
        Tuple of (parsed_position, full_response)
        - parsed_position: The extracted position as integer, or None if parsing failed
        - full_response: The complete unaltered response text
    """
    # Store the full response for logging
    full_response = response_text.strip()

    # Try multiple parsing strategies
    position = None

    # Strategy 1: Look for "Starting position:" followed by a number
    pattern1 = r"Starting position:\s*(\d+)"
    match1 = re.search(pattern1, response_text, re.IGNORECASE)
    if match1:
        position = int(match1.group(1))

    # Strategy 2: Look for "position" followed by a number
    if position is None:
        pattern2 = r"position\s+(?:is\s+)?(\d+)"
        match2 = re.search(pattern2, response_text, re.IGNORECASE)
        if match2:
            position = int(match2.group(1))

    # Strategy 3: Look for standalone numbers (be more careful here)
    if position is None:
        # Find all numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response_text)
        if numbers:
            # Take the first number that could be a valid position
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num < len(haystack):
                    position = num
                    break

    # Strategy 4: Look for the actual needle in the response and verify
    if position is not None:
        # Verify the parsed position is correct
        try:
            if position + len(needle) <= len(haystack):
                if haystack[position:position + len(needle)] == needle:
                    # Position is correct
                    pass
                else:
                    # Position is wrong, set to None
                    position = None
        except (IndexError, TypeError):
            position = None

    return position, full_response


def query_teacher_model(
    haystack: str,
    needle: str,
    model_name: str = "gemma3-27b-it-qat:latest",
    max_retries: int = 3,
    timeout_seconds: int = 60
) -> Dict[str, Any]:
    """
    Query the teacher model with a (haystack, needle) pair and get the response.

    Args:
        haystack: The full sequence containing the needle
        needle: The needle string to find
        model_name: Name of the Ollama model to use
        max_retries: Maximum number of retry attempts
        timeout_seconds: Timeout for each query attempt

    Returns:
        Dictionary containing:
        - 'success': Boolean indicating if query was successful
        - 'position': Parsed position (int) or None if parsing failed
        - 'teacher_full_response': Complete response from teacher
        - 'true_position': The actual correct position
        - 'is_correct': Boolean indicating if teacher's answer was correct
        - 'error': Error message if query failed
        - 'timestamp': When the query was made
        - 'model_used': Which model was actually used
    """
    # Find the true position for verification
    true_position = haystack.find(needle)
    if true_position == -1:
        return {
            'success': False,
            'position': None,
            'teacher_full_response': '',
            'true_position': -1,
            'is_correct': False,
            'error': 'Needle not found in haystack',
            'timestamp': datetime.now().isoformat(),
            'model_used': model_name
        }

    # Format the prompt
    prompt = format_needle_haystack_prompt(haystack, needle)

    # Try querying the model with retries
    for attempt in range(max_retries):
        try:
            print(f"Querying teacher model (attempt {attempt + 1}/{max_retries})...")

            # Make the query to Ollama
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for more consistent responses
                    'top_p': 0.9,
                    'num_predict': 200,  # Limit response length
                }
            )

            # Extract the response text
            response_text = response.get('response', '')

            if not response_text:
                raise ValueError("Empty response from model")

            # Parse the response
            parsed_position, full_response = parse_teacher_response(response_text, needle, haystack)

            # Check if parsing was successful
            is_correct = (parsed_position is not None and parsed_position == true_position)

            return {
                'success': True,
                'position': parsed_position,
                'teacher_full_response': full_response,
                'true_position': true_position,
                'is_correct': is_correct,
                'error': None,
                'timestamp': datetime.now().isoformat(),
                'model_used': model_name
            }

        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            print(f"Error: {error_msg}")

            if attempt == max_retries - 1:
                # Last attempt failed
                return {
                    'success': False,
                    'position': None,
                    'teacher_full_response': '',
                    'true_position': true_position,
                    'is_correct': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'model_used': model_name
                }

            # Wait before retrying
            time.sleep(2 ** attempt)  # Exponential backoff

    # Should never reach here, but just in case
    return {
        'success': False,
        'position': None,
        'teacher_full_response': '',
        'true_position': true_position,
        'is_correct': False,
        'error': 'All retry attempts failed',
        'timestamp': datetime.now().isoformat(),
        'model_used': model_name
    }


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    vocabulary = create_default_vocabulary()
    needle = "TARGET"
    haystack_len = 100

    print("Testing basic needle-in-haystack generation:")
    sequence, needle_found, position = generate_needle_haystack_task(
        haystack_len=haystack_len,
        needle_string=needle,
        vocabulary=vocabulary,
        random_seed=42
    )

    print(f"Haystack length: {len(sequence)}")
    print(f"Needle: '{needle_found}'")
    print(f"Position: {position}")
    print(f"Verification: '{sequence[position:position+len(needle)]}'")
    print(f"Match: {sequence[position:position+len(needle)] == needle}")
    print()

    # Test batch generation
    print("Testing batch generation:")
    sequences, needles, positions = generate_needle_haystack_batch(
        batch_size=3,
        haystack_len=50,
        needle_string="FIND",
        vocabulary=vocabulary,
        random_seed=123
    )

    for i, (seq, needle, pos) in enumerate(zip(sequences, needles, positions)):
        print(f"Batch {i+1}: Position {pos}, Needle '{needle}', Length {len(seq)}")
        print(f"  Verification: '{seq[pos:pos+len(needle)]}' == '{needle}': {seq[pos:pos+len(needle)] == needle}")

    print("\n" + "="*50)
    print("Testing teacher model querying pipeline:")
    print("="*50)

    # Test teacher querying with a simple example
    test_haystack = "abcdefghijklmnopqrstuvwxyzTARGETzyxwvutsrqponmlkjihgfedcba"
    test_needle = "TARGET"

    print(f"Test haystack: {test_haystack}")
    print(f"Test needle: {test_needle}")
    print(f"True position: {test_haystack.find(test_needle)}")
    print()

    # Test prompt formatting
    print("Testing prompt formatting:")
    prompt = format_needle_haystack_prompt(test_haystack, test_needle)
    print("Generated prompt:")
    print("-" * 30)
    print(prompt)
    print("-" * 30)
    print()

    # Test response parsing with mock responses
    print("Testing response parsing:")
    mock_responses = [
        "The substring 'TARGET' starts at position 26. Starting position: 26",
        "I found the substring at position 26 in the sequence.",
        "Starting position: 26\n\nThe substring TARGET appears at index 26.",
        "The answer is 26.",
        "Position is 26"
    ]

    for i, mock_response in enumerate(mock_responses):
        parsed_pos, full_resp = parse_teacher_response(mock_response, test_needle, test_haystack)
        print(f"Mock response {i+1}: '{mock_response}'")
        print(f"  Parsed position: {parsed_pos}")
        print(f"  Correct: {parsed_pos == test_haystack.find(test_needle)}")
        print()

    # Optionally test actual teacher model (commented out by default)
    print("To test the actual teacher model, uncomment the following code:")
    print("# result = query_teacher_model(test_haystack, test_needle)")
    print("# print('Teacher model result:', result)")

    print("\nAll tests completed successfully!")
