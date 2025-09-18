import tensorflow as tf
import random
import json
import os

def read_congruent_tfrecord(filepath):
    """Read congruent tfrecord and extract all information."""
    raw_dataset = tf.data.TFRecordDataset(filepath)
    
    examples = []
    for raw_record in raw_dataset.take(1000):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # Extract all fields
        record_data = {
            'question': example.features.feature['question'].bytes_list.value[0].decode('utf-8'),
            'answer': example.features.feature['answer'].bytes_list.value[0].decode('utf-8'),
            'metadata': example.features.feature['metadata'].bytes_list.value[0].decode('utf-8')
        }
        
        # Parse metadata to get entities and distance
        metadata = eval(record_data['metadata'])  # Convert string dict to actual dict
        record_data['entities'] = eval(metadata['entities'])  # Convert string list to actual list
        record_data['distance'] = metadata['distance']
        
        examples.append(record_data)
    
    return examples

def permute_relationships(question_text, entities, permutation_map):
    """Apply permutation to the relationships in the question."""
    permuted_question = question_text
    
    # Replace each entity with its permuted version
    # Do this carefully to avoid partial replacements
    # First, replace with temporary placeholders
    for i, original in enumerate(entities):
        permuted_question = permuted_question.replace(original, f"__TEMP_{i}__")
    
    # Then replace placeholders with permuted entities
    for i, original in enumerate(entities):
        permuted_entity = permutation_map[original]
        permuted_question = permuted_question.replace(f"__TEMP_{i}__", permuted_entity)
    
    return permuted_question

def create_permuted_dataset(input_path, output_path):
    """Create permuted version of congruent dataset."""
    
    # Read all examples
    print("Reading congruent dataset...")
    examples = read_congruent_tfrecord(input_path)
    print(f"Found {len(examples)} examples")
    
    # For each example, we'll create a random permutation of its entities
    permuted_examples = []
    permutation_records = []
    
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            print(f"Processing example {idx}...")
        
        entities = example['entities']
        
        # Create a random permutation
        permuted_entities = entities.copy()
        random.shuffle(permuted_entities)
        
        # Create permutation mapping
        permutation_map = {orig: perm for orig, perm in zip(entities, permuted_entities)}
        
        # Apply permutation to question text
        permuted_question = permute_relationships(example['question'], entities, permutation_map)
        
        # Determine new answer based on permuted relationships
        # Extract the query from the question (last line before "Answer yes or no")
        question_lines = example['question'].strip().split('\n')
        query_line = question_lines[-1]  # e.g., "Is X larger than Y?"
        
        # Apply permutation to query to check answer
        permuted_query = permute_relationships(query_line, entities, permutation_map)
        
        # Since we're permuting randomly, we need to recompute the answer
        # For now, we'll keep the same answer structure but you may need to verify
        # this based on the actual permuted relationships
        
        # Create new example
        permuted_example = {
            'question': permuted_question,
            'answer': example['answer'],  # This might need adjustment based on permutation
            'metadata': str({
                'distance': example['distance'],
                'entities': str(permuted_entities),
                'answer': example['answer'],
                'original_entities': str(entities),
                'permutation': str(permutation_map)
            })
        }
        
        permuted_examples.append(permuted_example)
        
        # Record permutation for reference
        permutation_records.append({
            'example_idx': idx,
            'original_entities': entities,
            'permuted_entities': permuted_entities,
            'permutation_map': permutation_map
        })
    
    # Write permuted dataset to tfrecord
    print(f"\nWriting permuted dataset to {output_path}...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for example in permuted_examples:
            # Create tf.Example
            feature = {
                'question': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['question'].encode()])),
                'answer': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['answer'].encode()])),
                'metadata': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['metadata'].encode()]))
            }
            
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
    
    # Save permutation records as JSON for reference
    permutation_file = output_path.replace('.tfrecord', '_permutations.json')
    print(f"Saving permutation records to {permutation_file}...")
    with open(permutation_file, 'w') as f:
        json.dump(permutation_records, f, indent=2)
    
    print(f"Successfully created permuted dataset with {len(permuted_examples)} examples")
    return permutation_records

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    input_path = 'congruent_incongruent_1000/test/comparison_congruent_size_test.tfrecord'
    output_path = 'congruent_incongruent_1000/test/comparison_permuted_size_test.tfrecord'
    
    # Create permuted dataset
    permutation_records = create_permuted_dataset(input_path, output_path)
    
    # Print some examples to verify
    print("\nExample permutations (first 3):")
    for record in permutation_records[:3]:
        print(f"Example {record['example_idx']}:")
        print(f"  Original: {record['original_entities'][:5]}...")
        print(f"  Permuted: {record['permuted_entities'][:5]}...")
        print()

if __name__ == "__main__":
    main()