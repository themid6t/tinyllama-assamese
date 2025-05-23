import os
import time
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import argparse
import json
import sys
from dotenv import load_dotenv
import random
from collections import defaultdict

load_dotenv()

class TranslationPipeline:
    def __init__(self, api_key=None, model="gpt-4o", batch_size=25):
        """
        Initialize the translation pipeline.
        
        Args:
            api_key: OpenAI API key. If None, will try to get it from environment variable.
            model: OpenAI model to use for translation
            batch_size: Number of samples to translate in a single API call
        """
        # Initialize OpenAI client with the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        print(f"Initialized translation pipeline with model: {model}")
    
    def translate_batch(self, texts, source_lang="English", target_lang="Assamese"):
        """
        Translate a batch of texts from source language to target language.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language
        
        Returns:
            List of translated texts
        """
        # Prepare the content for the API call
        messages = [
            {
                "role": "system",
                "content": f"You are a professional translator. Translate the following texts from {source_lang} to {target_lang}. Return ONLY the translations in a JSON structure with a 'translations' key containing an array of translated strings."
            },
            {
                "role": "user",
                "content": f"Translate these texts to {target_lang}:\n{json.dumps(texts)}"
            }
        ]
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extract the translated texts from the response
            response_content = response.choices[0].message.content
            print(f"API Response: {response_content[:100]}...")
            
            # Parse the JSON response
            try:
                response_json = json.loads(response_content)
                if "translations" in response_json:
                    translated_texts = response_json["translations"]
                else:
                    # Try to find any array in the response
                    for key, value in response_json.items():
                        if isinstance(value, list) and len(value) > 0:
                            translated_texts = value
                            print(f"Found translations under key: '{key}'")
                            break
                    else:
                        # If no suitable array found, use the first value if it's a string
                        if len(response_json) > 0:
                            first_value = next(iter(response_json.values()))
                            if isinstance(first_value, str):
                                translated_texts = [first_value]
                            else:
                                raise ValueError(f"Could not find translations in response: {response_content}")
                        else:
                            raise ValueError(f"Empty response JSON")
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract text directly
                print("JSON parsing failed, attempting to extract translations directly...")
                translated_texts = [response_content.strip()]
            
            # Ensure correct number of translations
            if len(translated_texts) != len(texts):
                print(f"Warning: Expected {len(texts)} translations, but got {len(translated_texts)}")
                # If we got fewer translations than expected, try extracting individual translations
                if len(translated_texts) < len(texts):
                    # Try to fill missing translations with empty strings
                    translated_texts.extend(["" for _ in range(len(texts) - len(translated_texts))])
                else:
                    # If we got more translations than expected, truncate
                    translated_texts = translated_texts[:len(texts)]
            
            return translated_texts
            
        except Exception as e:
            print(f"Error in translation: {str(e)}")
            print(f"Failed to translate texts. Will retry with simpler approach.")
            
            # Simpler fallback approach - translate one by one
            translated_texts = []
            for text in texts:
                try:
                    single_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": f"Translate this text from {source_lang} to {target_lang}. Return ONLY the translation."},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.3
                    )
                    translation = single_response.choices[0].message.content.strip()
                    translated_texts.append(translation)
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                except Exception as inner_e:
                    print(f"Error translating single text: {str(inner_e)}")
                    translated_texts.append("")
            
            return translated_texts

def load_oasst1_english_samples(num_samples=2000, seed=42):
    """
    Load OASST1 dataset and select complete conversation threads in English.
    Captures ALL responses for each prompt, not just one conversation path.
    
    Args:
        num_samples: Target number of samples to select (approximate)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the selected English samples with preserved conversation structure
    """
    print(f"Loading OASST1 dataset and selecting conversation threads (target: ~{num_samples} samples)...")
    
    # Load the Open Assistant Conversations Dataset
    dataset = load_dataset("OpenAssistant/oasst1")
    
    # Get the training split
    train_data = dataset["train"]
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Filter for English messages only
    english_data = [item for item in train_data if item["lang"] == "en"]
    total_english = len(english_data)
    print(f"Total English samples found: {total_english}")
    
    # Create lookup dictionaries for efficient access
    message_lookup = {item["message_id"]: item for item in english_data}
    
    # Group messages by parent_id to find children more efficiently
    children_map = defaultdict(list)
    for item in english_data:
        if item["parent_id"] is not None and item["parent_id"] in message_lookup:
            children_map[item["parent_id"]].append(item["message_id"])
    
    # Find all root messages (no parent_id)
    root_messages = [item for item in english_data if item["parent_id"] is None]
    print(f"Found {len(root_messages)} root messages")
    
    # Function to recursively collect ALL branches in a conversation tree
    def collect_conversation_tree(root_id):
        """
        Collect all messages in a conversation tree including all branches.
        Returns a list of messages in the conversation.
        """
        if root_id not in message_lookup:
            return []
            
        thread = [message_lookup[root_id]]
        
        # Get all direct children of this message
        child_ids = children_map.get(root_id, [])
        
        # Process each child and their branches
        for child_id in child_ids:
            child_branch = collect_conversation_tree(child_id)
            thread.extend(child_branch)
            
        return thread
    
    # Count branches in conversation trees
    def count_branches(root_id, depth=0):
        """Count branches in conversation tree"""
        branches = 0
        child_ids = children_map.get(root_id, [])
        
        if not child_ids:  # Leaf node
            return 1
            
        for child_id in child_ids:
            branches += count_branches(child_id, depth+1)
            
        return branches
    
    # Analyze and collect complete conversation trees with branching info
    print("Building complete conversation trees (including all branches)...")
    
    conversation_stats = []
    for root in root_messages:
        root_id = root["message_id"]
        branch_count = count_branches(root_id)
        msg_count = len(collect_conversation_tree(root_id))
        conversation_stats.append((root_id, branch_count, msg_count))
    
    # Sort by number of messages, focusing on rich conversations
    conversation_stats.sort(key=lambda x: x[2], reverse=True)
    
    # Print statistics about conversation branching
    branch_counts = [stat[1] for stat in conversation_stats]
    msg_counts = [stat[2] for stat in conversation_stats]
    if branch_counts:
        print(f"Average branches per conversation: {sum(branch_counts) / len(branch_counts):.2f}")
        print(f"Max branches: {max(branch_counts)}")
        print(f"Average messages per conversation: {sum(msg_counts) / len(msg_counts):.2f}")
        print(f"Conversations with multiple branches: {sum(1 for b in branch_counts if b > 1)}")
    
    # Collect complete conversation trees with ALL branches
    all_conversations = []
    for root_id, branch_count, _ in conversation_stats:
        # Only include conversations with at least one response
        if branch_count >= 1:
            thread = collect_conversation_tree(root_id)
            all_conversations.append(thread)
    
    print(f"Collected {len(all_conversations)} complete conversations with all branches")
    
    # Shuffle conversations for random selection while maintaining tree structure
    random.shuffle(all_conversations)
    
    # Select whole conversations until we reach or exceed the target count
    selected_messages = []
    current_count = 0
    selected_conv_count = 0
    
    # For very small sample sizes, make sure we select at least one conversation
    min_conversations_to_select = 1
    
    for conv in all_conversations:
        if selected_conv_count < min_conversations_to_select or current_count < num_samples:
            selected_messages.extend(conv)
            current_count += len(conv)
            selected_conv_count += 1
        else:
            break
    
    print(f"Selected {selected_conv_count} conversations with a total of {len(selected_messages)} messages")
    
    # Convert to DataFrame
    df = pd.DataFrame(selected_messages)
    
    # If we've selected too many samples, truncate at conversation boundaries,
    # but always keep at least one complete conversation
    if len(df) > num_samples * 1.5 and len(all_conversations) > 1:
        print(f"Selected too many samples ({len(df)}), truncating to approximately {num_samples}...")
        
        # Recalculate how many complete conversations we can include
        truncated_messages = []
        messages_so_far = 0
        truncated_conv_count = 0
        
        for conv in all_conversations[:selected_conv_count]:
            # Always include at least the first conversation
            if truncated_conv_count == 0 or messages_so_far + len(conv) <= num_samples * 1.2:
                truncated_messages.extend(conv)
                messages_so_far += len(conv)
                truncated_conv_count += 1
            else:
                break
                
        if truncated_messages:  # Check that we have at least some messages
            df = pd.DataFrame(truncated_messages)
            print(f"Truncated to {len(df)} samples from {truncated_conv_count} complete conversations")
    
    # Verify we have at least some data
    if len(df) == 0 and all_conversations:
        # If we somehow ended up with 0 samples but have conversations available,
        # just take the first conversation no matter its size
        first_conv = all_conversations[0]
        df = pd.DataFrame(first_conv)
        print(f"Ensuring at least one conversation: using {len(df)} samples from 1 conversation")
    
    return df

def inspect_conversations(df):
    """
    Print a summary of the conversations in the DataFrame.
    Shows ALL branches in each conversation, including multiple assistant responses.
    
    Args:
        df: DataFrame containing conversation messages
    """
    if 'parent_id' not in df.columns or 'message_id' not in df.columns:
        print("DataFrame does not have required columns for conversation inspection")
        return
        
    # Find all root messages
    root_messages = df[df['parent_id'].isna()]
    print(f"Found {len(root_messages)} root messages in the dataset")
    
    # Build parent-child relationships for efficient traversal
    children_map = defaultdict(list)
    for _, row in df.iterrows():
        if not pd.isna(row['parent_id']):
            children_map[row['parent_id']].append(row['message_id'])
    
    # Count branches and messages per conversation
    conversation_sizes = []
    branching_factors = []
    
    for _, root in root_messages.iterrows():
        root_id = root['message_id']
        
        # Count all messages in this conversation tree
        def count_messages_and_branches(msg_id):
            """Count messages and branches in a conversation tree"""
            msgs = 1  # Count this message
            branches = 0
            children = children_map.get(msg_id, [])
            
            if not children:  # Leaf node
                return msgs, 0
            
            # If more than one child, we have branches
            if len(children) > 1:
                branches = len(children)
            
            # Recursively count for all children
            for child_id in children:
                child_msgs, child_branches = count_messages_and_branches(child_id)
                msgs += child_msgs
                branches += child_branches
                
            return msgs, branches
        
        msg_count, branch_count = count_messages_and_branches(root_id)
        conversation_sizes.append(msg_count)
        branching_factors.append(branch_count)
    
    if conversation_sizes:
        print(f"Average conversation size: {sum(conversation_sizes)/len(conversation_sizes):.2f} messages")
        print(f"Max conversation size: {max(conversation_sizes)} messages")
        print(f"Min conversation size: {min(conversation_sizes)} messages")
        
        print(f"Average branches per conversation: {sum(branching_factors)/len(branching_factors):.2f}")
        print(f"Max branches: {max(branching_factors)} in a conversation")
        print(f"Conversations with multiple branches: {sum(1 for b in branching_factors if b > 0)}")
        
        # Show distribution of conversation sizes
        from collections import Counter
        size_counts = Counter(conversation_sizes)
        print("\nConversation size distribution:")
        for size in sorted(size_counts.keys()):
            print(f"{size} messages: {size_counts[size]} conversations")
    
    # Sample and show a few full conversations with all branches
    print("\nSample conversations (with ALL branches):")
    sample_size = min(3, len(root_messages))
    sampled_roots = root_messages.sample(sample_size)
    
    for i, (_, root) in enumerate(sampled_roots.iterrows(), 1):
        root_id = root['message_id']
        print(f"\nConversation {i} (tree view with all branches):")
        
        # Use a tree structure for better visualization
        def print_message_tree(msg_id, depth=0, branch_idx=None):
            """Print a message and ALL of its children recursively"""
            try:
                msg = df[df['message_id'] == msg_id].iloc[0]
                prefix = "  " * depth
                branch_marker = f"[Branch {branch_idx}] " if branch_idx is not None and depth > 0 else ""
                role = msg['role']
                text = msg['text']
                print(f"{prefix}{branch_marker}[{role}]: {text[:100]}{'...' if len(text) > 100 else ''}")
                
                # Print all children with branch indicators when needed
                children = children_map.get(msg_id, [])
                
                if len(children) > 1:
                    # Multiple branches - number them
                    for idx, child_id in enumerate(children, 1):
                        print_message_tree(child_id, depth + 1, idx)
                elif children:
                    # Single child - continue the current branch
                    print_message_tree(children[0], depth + 1, None)
                    
            except IndexError:
                print(f"{prefix}Message ID {msg_id} not found in dataframe")
        
        print_message_tree(root_id)

def load_existing_or_create_new(output_path, num_samples):
    """
    Load existing translation file or create a new one.
    
    Args:
        output_path: Path to the output CSV file
        num_samples: Number of samples to select if creating new file
        
    Returns:
        DataFrame containing the data
    """
    if os.path.exists(output_path):
        print(f"Found existing file: {output_path}")
        df = pd.read_csv(output_path)
        print(f"Loaded {len(df)} existing translations")
        return df
    else:
        print(f"No existing file found, creating new dataset with {num_samples} samples")
        return load_oasst1_english_samples(num_samples=num_samples)

def translate_and_save(df, output_path, translator, batch_size=25, save_interval=100):
    """
    Translate English samples to Assamese and save results.
    
    Args:
        df: DataFrame containing English samples
        output_path: Path to save the output CSV
        translator: TranslationPipeline instance
        batch_size: Number of samples to translate in one API call
        save_interval: Save after processing this many samples
    """
    # Check if translation column already exists
    if 'assamese_translation' not in df.columns:
        df['assamese_translation'] = ""
    
    # Count how many translations are already done
    done_count = df[df['assamese_translation'] != ""].shape[0]
    if done_count > 0:
        print(f"Found {done_count} existing translations")
    
    # Process the untranslated items
    total = len(df)
    
    # Create batches of texts to translate
    batches = []
    batch = []
    batch_indices = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.isna(row.get('assamese_translation')) or row['assamese_translation'] == "":
            batch.append(row['text'])
            batch_indices.append(idx)
            
            if len(batch) >= batch_size:
                batches.append((batch, batch_indices))
                batch = []
                batch_indices = []
    
    # Add the last batch if it's not empty
    if batch:
        batches.append((batch, batch_indices))
    
    print(f"Created {len(batches)} batches for translation")
    
    # Translate batch by batch and save at intervals
    for i, (texts, indices) in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(texts)} texts)...")
        
        # Translate
        translations = translator.translate_batch(texts)
        
        # Update dataframe
        for j, idx in enumerate(indices):
            df.at[idx, 'assamese_translation'] = translations[j]
        
        # Save at intervals or ask for continuation
        current_count = done_count + (i + 1) * batch_size
        if (i + 1) % (save_interval // batch_size) == 0 or i == len(batches) - 1:
            df.to_csv(output_path, index=False)
            print(f"Saved progress to {output_path} ({min(current_count, total)}/{total} samples processed)")
            
            # Ask whether to continue
            if i < len(batches) - 1:
                while True:
                    response = input("Continue translation? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        break
                    elif response in ['n', 'no']:
                        return
                    else:
                        print("Please enter 'y' or 'n'")
    
    print(f"Completed translation of {total} samples")
    df.to_csv(output_path, index=False)
    return df

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Translate OASST1 English samples to Assamese")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--samples", type=int, default=2000, help="Target number of samples to select")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for translation")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval")
    parser.add_argument("--output", type=str, default="oasst1_english_assamese.csv", help="Output file name")
    parser.add_argument("--inspect", action="store_true", help="Inspect conversations in detail")
    args = parser.parse_args()
    
    # Set output path
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, args.output)
    
    # Load or create dataset
    df = load_existing_or_create_new(output_path, args.samples)
    
    # Inspect conversations if requested
    if args.inspect:
        inspect_conversations(df)
        if input("Continue with translation? (y/n): ").lower() != 'y':
            return
    
    # Initialize translator
    translator = TranslationPipeline(api_key=args.api_key, model=args.model, batch_size=args.batch_size)
    
    # Main processing loop
    while True:
        # Translate and save
        translate_and_save(df, output_path, translator, 
                          batch_size=args.batch_size, 
                          save_interval=args.save_interval)
        
        # Check if we want to translate more samples
        while True:
            response = input("Do you want to translate more samples? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                more_samples = int(input("How many more samples do you want to translate? "))
                # Get more samples and add to the dataframe
                more_df = load_oasst1_english_samples(num_samples=more_samples, 
                                                     seed=int(time.time()))
                # Avoid duplicates by checking message_id
                existing_ids = set(df['message_id'])
                more_df = more_df[~more_df['message_id'].isin(existing_ids)]
                if len(more_df) > 0:
                    df = pd.concat([df, more_df], ignore_index=True)
                    print(f"Added {len(more_df)} new samples to translate")
                else:
                    print("No new samples could be added (all were duplicates)")
                break
            elif response in ['n', 'no']:
                print(f"Translation complete. Output saved to: {output_path}")
                return
            else:
                print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)