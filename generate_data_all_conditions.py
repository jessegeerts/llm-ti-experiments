import tensorflow as tf
import os
from tqdm import tqdm

def read_tfrecord(filepath):
    """Read tfrecord file and extract questions."""
    raw_dataset = tf.data.TFRecordDataset(filepath)
    
    questions = []
    for raw_record in raw_dataset.take(1000):  # Take 1000 examples
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # Extract the question field
        question = example.features.feature['question'].bytes_list.value[0].decode('utf-8')
        question = question.replace('\n', '\\n') 
        questions.append(question)
    
    return questions


def create_prompt(base_text, condition):
    """Apply different prompting conditions to the base question."""
    
    # Geometry conditions
    if condition == 'baseline':
        return f"{base_text}\\n\\nAnswer yes or no."
    elif condition == 'numberline':
        return f"Imagine all of these items lie on a number line from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'road':
        return f"Imagine all of these items lie along a road from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'circle':
        return f"Imagine all of these items lie on a circle from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'clusters':
        return f"Imagine all of these items lie in separate clusters from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == '3d_space':
        return f"Imagine all of these items lie scattered in 3D space from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'cloud':
        return f"Imagine all of these items lie in a cloud from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'hyperbolic':
        return f"Imagine all of these items lie on a hyperbolic plane from smallest to largest.\\n\\n{base_text}\\n\\nAnswer yes or no."
    elif condition == 'cot':
        return f"Let's think step by step to trace through the relationships and determine the answer.\\n\\n{base_text}\\n\\nAnswer yes or no."
    else:
        raise ValueError(f"Unknown condition: {condition}")

def main():
    # Define paths to your tfrecord files
    tfrecord_paths = {
        'congruent': 'congruent_incongruent_1000/test/comparison_congruent_size_test.tfrecord',
        'incongruent': 'congruent_incongruent_1000/test/comparison_incongruent_size_test.tfrecord',
        'random': 'congruent_incongruent_1000/test/comparison_random_string_size_test.tfrecord',
        'permuted': 'congruent_incongruent_1000/test/comparison_permuted_size_test.tfrecord'
    }
    
    # Define all conditions
    prompt_conditions = [
        'baseline',
        'numberline',
        'road',
        'circle',
        'clusters',
        '3d_space',
        'cloud',
        'hyperbolic',
        'cot'
    ]
    
    # Output file
    output_file = 'all_prompts.txt'
    
    # Generate all prompts
    all_prompts = []
    
    for item_type in ['congruent', 'incongruent', 'random', 'permuted']:
        print(f"Processing {item_type} items...")
        
        # Read questions from tfrecord
        questions = read_tfrecord(tfrecord_paths[item_type])
        print(f"  Found {len(questions)} questions")
        
        for condition in prompt_conditions:
            print(f"  Applying {condition} condition...")
            
            for question in questions:
                prompt = create_prompt(question, condition)
                # Add metadata as a comment (optional - can remove if not needed)
                prompt_with_metadata = f"# {item_type}_{condition}\n{prompt}"
                all_prompts.append(prompt)
    
    # Write to file
    print("\nWriting prompts to file...")
    with open(output_file, 'w') as f:
        for prompt in tqdm(all_prompts, desc="Writing to file"):
            prompt_separator = '\n\n---\n\n'
            f.write(prompt + prompt_separator)  # Separator between prompts
    
    print(f"\nGenerated {len(all_prompts)} total prompts")
    print(f"Saved to {output_file}")
    
    # Verify counts
    expected_count = 4 * 9 * 1000  # 3 item types × 9 conditions × 1000 examples
    print(f"Expected: {expected_count}, Got: {len(all_prompts)}")

if __name__ == "__main__":
    main()