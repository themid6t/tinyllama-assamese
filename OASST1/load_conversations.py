#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OASST1 Assamese Dataset Loader Script
This script loads translated conversations from the OASST1 English-Assamese dataset,
organizes them into conversation threads, and saves them in various formats.
"""

import os
import json
import pandas as pd
import argparse
from collections import defaultdict
import random
import re


def clean_text(text):
    """Clean text by removing extra whitespaces and normalizing quotes."""
    if pd.isna(text) or text is None or text == "":
        return ""
    # Normalize different types of quotes to standard quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace
    return text.strip()


def build_conversation_threads(df):
    """
    Build complete conversation threads from dataframe, preserving all branching.
    
    Args:
        df: DataFrame with message_id, parent_id, text, assamese_translation, role
        
    Returns:
        List of conversation threads, each containing a list of messages in order
    """
    # Check required columns
    required_cols = ['message_id', 'parent_id', 'text', 'assamese_translation', 'role']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
    # Create lookup maps
    message_lookup = {
        row['message_id']: {
            'id': row['message_id'],
            'parent_id': row['parent_id'],
            'text': clean_text(row['text']),
            'translation': clean_text(row['assamese_translation']),
            'role': row['role']
        }
        for _, row in df.iterrows() if not pd.isna(row['message_id'])
    }
    
    # Build parent-child relationships
    children_map = defaultdict(list)
    for msg_id, msg in message_lookup.items():
        parent_id = msg['parent_id']
        if not pd.isna(parent_id) and parent_id in message_lookup:
            children_map[parent_id].append(msg_id)
    
    # Find all root messages (no parent_id)
    root_messages = [msg_id for msg_id, msg in message_lookup.items() 
                     if pd.isna(msg['parent_id'])]
    
    print(f"Found {len(root_messages)} root messages")
    
    # Build conversation trees
    all_conversations = []
    
    def collect_conversation_thread(msg_id, thread=None):
        """Recursively collect a conversation thread with all messages."""
        if thread is None:
            thread = []
            
        msg = message_lookup[msg_id]
        thread.append(msg)
        
        # Process all children in order
        for child_id in children_map.get(msg_id, []):
            collect_conversation_thread(child_id, thread)
            
        return thread
    
    # Process each root to create complete conversations
    skipped = 0
    for root_id in root_messages:
        try:
            conversation = collect_conversation_thread(root_id)
            
            # Verify that the conversation has translations
            if all(msg.get('translation') for msg in conversation):
                # Organize conversations as alternating human/assistant messages
                if conversation[0]['role'] == 'prompter':
                    all_conversations.append(conversation)
                else:
                    skipped += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing conversation {root_id}: {e}")
            skipped += 1
    
    print(f"Built {len(all_conversations)} complete conversations, skipped {skipped}")
    
    return all_conversations


def save_conversations_jsonl(conversations, output_path):
    """Save conversations in JSONL format with alternating human/assistant roles."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            # Format as {"messages": [{"role": "...", "content": "..."}, ...]}
            formatted_conv = {
                "messages": []
            }
            
            current_idx = 0
            while current_idx < len(conv):
                # Process human message
                if current_idx < len(conv) and conv[current_idx]['role'] == 'prompter':
                    formatted_conv["messages"].append({
                        "role": "user", 
                        "content": conv[current_idx]['translation']
                    })
                    current_idx += 1
                
                # Process assistant message
                if current_idx < len(conv) and conv[current_idx]['role'] == 'assistant':
                    formatted_conv["messages"].append({
                        "role": "assistant", 
                        "content": conv[current_idx]['translation']
                    })
                    current_idx += 1
            
            # Only save if we have at least one exchange
            if len(formatted_conv["messages"]) >= 2:
                json.dump(formatted_conv, f, ensure_ascii=False)
                f.write('\n')


def save_conversations_text(conversations, output_path):
    """Save conversations in a human-readable text format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, conv in enumerate(conversations):
            f.write(f"=== Conversation {i+1} ===\n\n")
            
            for msg in conv:
                role_name = "Human" if msg['role'] == 'prompter' else "Assistant"
                f.write(f"{role_name}: {msg['translation']}\n\n")
            
            f.write("\n" + "="*50 + "\n\n")


def save_conversations_side_by_side(conversations, output_path):
    """Save conversations in a side-by-side format with English and Assamese."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, conv in enumerate(conversations):
            f.write(f"=== Conversation {i+1} ===\n\n")
            
            for msg in conv:
                role_name = "Human" if msg['role'] == 'prompter' else "Assistant"
                f.write(f"{role_name} (English): {msg['text']}\n")
                f.write(f"{role_name} (Assamese): {msg['translation']}\n\n")
            
            f.write("\n" + "="*50 + "\n\n")


def main():
    """Main function to load and process the translated conversations."""
    parser = argparse.ArgumentParser(description="Load and process translated OASST1 conversations")
    parser.add_argument("--input", type=str, default="oasst1_english_assamese.csv", 
                        help="Input CSV file with translations")
    parser.add_argument("--output_dir", type=str, default="conversations",
                        help="Directory to save the processed conversations")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of random conversations to sample (default: use all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Set input and output paths
    input_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(input_dir, args.input)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the translated dataset
    print(f"Loading translations from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} translated messages")
    
    # Build conversation threads
    conversations = build_conversation_threads(df)
    
    # Sample conversations if requested
    if args.sample is not None and args.sample < len(conversations):
        random.seed(args.seed)
        conversations = random.sample(conversations, args.sample)
        print(f"Sampled {args.sample} random conversations")
    
    # Save in different formats
    # 1. JSONL format for fine-tuning
    jsonl_path = os.path.join(output_dir, "conversations.jsonl")
    save_conversations_jsonl(conversations, jsonl_path)
    print(f"Saved {len(conversations)} conversations to {jsonl_path}")
    
    # 2. Human-readable text format
    text_path = os.path.join(output_dir, "conversations.txt")
    save_conversations_text(conversations, text_path)
    print(f"Saved human-readable conversations to {text_path}")
    
    # 3. Side-by-side English and Assamese
    parallel_path = os.path.join(output_dir, "conversations_bilingual.txt")
    save_conversations_side_by_side(conversations, parallel_path)
    print(f"Saved bilingual conversations to {parallel_path}")
    
    # 4. Python pickle for programmatic access
    import pickle
    pickle_path = os.path.join(output_dir, "conversations.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(conversations, f)
    print(f"Saved Python pickle with all conversations to {pickle_path}")
    
    print("\nConversation Statistics:")
    print(f"Total conversations: {len(conversations)}")
    
    # Count total messages
    total_messages = sum(len(conv) for conv in conversations)
    print(f"Total messages: {total_messages}")
    
    # Count messages by role
    human_msgs = sum(sum(1 for msg in conv if msg['role'] == 'prompter') for conv in conversations)
    assistant_msgs = sum(sum(1 for msg in conv if msg['role'] == 'assistant') for conv in conversations)
    print(f"Human messages: {human_msgs}")
    print(f"Assistant messages: {assistant_msgs}")
    
    # Conversation length distribution
    lengths = [len(conv) for conv in conversations]
    print(f"Average conversation length: {sum(lengths)/len(lengths):.2f} messages")
    print(f"Min conversation length: {min(lengths)} messages")
    print(f"Max conversation length: {max(lengths)} messages")
    
    print("\nDone! Your conversations are ready for fine-tuning.")


if __name__ == "__main__":
    main()