#!/usr/bin/env python3
"""
Dataset Loaders for MARU Training

This module provides unified loaders for various datasets that test MARU's
core capabilities: linear scaling, persistent memory, and long-range dependencies.
"""

import json
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from pathlib import Path


class MARUDatasetLoader:
    """Unified loader for MARU-optimized datasets."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_msc_dataset(self, split: str = "train") -> List[Dict[str, Any]]:
        """Load Multi-Session Chat dataset for persistent memory testing."""
        print("📥 Loading MSC (Multi-Session Chat) dataset...")
        
        try:
            dataset = load_dataset("MemGPT/MSC-Self-Instruct", split=split, cache_dir=self.cache_dir)
            
            formatted_data = []
            for example in dataset:
                formatted_data.append({
                    'task_type': 'multi_session_memory',
                    'session_data': example,
                    'memory_test': True,
                    'sequence_length': len(str(example).split())
                })
            
            print(f"✅ Loaded {len(formatted_data)} MSC examples")
            return formatted_data
            
        except Exception as e:
            print(f"❌ Failed to load MSC dataset: {e}")
            return []
    
    def load_narrativeqa_dataset(self, split: str = "train", max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load NarrativeQA dataset for long document comprehension."""
        print("📥 Loading NarrativeQA dataset...")
        
        try:
            dataset = load_dataset("deepmind/narrativeqa", split=split, cache_dir=self.cache_dir)
            
            formatted_data = []
            count = 0
            
            for example in dataset:
                if max_examples and count >= max_examples:
                    break
                    
                # Extract document text
                document_text = example['document']['text']
                doc_length = len(document_text.split())
                
                # Only include documents that test MARU's scaling (2K+ tokens)
                if doc_length >= 2000:
                    formatted_data.append({
                        'task_type': 'long_document_qa',
                        'document': document_text,
                        'question': example['question']['text'],
                        'answers': [ans['text'] for ans in example['answers']],
                        'document_length': doc_length,
                        'story_id': example['document']['id']
                    })
                    count += 1
            
            print(f"✅ Loaded {len(formatted_data)} NarrativeQA examples (2K+ tokens)")
            return formatted_data
            
        except Exception as e:
            print(f"❌ Failed to load NarrativeQA dataset: {e}")
            return []
    
    def load_booksum_dataset(self, split: str = "train", max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load BookSum dataset for book-length summarization."""
        print("📥 Loading BookSum dataset...")
        
        try:
            dataset = load_dataset("ubaada/booksum-complete-cleaned", split=split, cache_dir=self.cache_dir)
            
            formatted_data = []
            count = 0
            
            for example in dataset:
                if max_examples and count >= max_examples:
                    break
                    
                doc_length = len(example['text'].split())
                
                # Focus on longer documents that challenge transformers
                if doc_length >= 3000:
                    formatted_data.append({
                        'task_type': 'long_document_summarization',
                        'document': example['text'],
                        'summary': example['summary'],
                        'document_length': doc_length,
                        'book_title': example.get('title', 'Unknown'),
                        'chapter_title': example.get('chapter_title', 'Unknown')
                    })
                    count += 1
            
            print(f"✅ Loaded {len(formatted_data)} BookSum examples (3K+ tokens)")
            return formatted_data
            
        except Exception as e:
            print(f"❌ Failed to load BookSum dataset: {e}")
            return []
    
    def load_lambada_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load LAMBADA dataset for long-range dependency testing."""
        print("📥 Loading LAMBADA dataset...")
        
        try:
            dataset = load_dataset("EleutherAI/lambada_openai", split=split, cache_dir=self.cache_dir)
            
            formatted_data = []
            for example in dataset:
                context_length = len(example['text'].split())
                
                formatted_data.append({
                    'task_type': 'long_range_prediction',
                    'context': example['text'],
                    'target_word': example['text'].split()[-1],
                    'context_without_target': ' '.join(example['text'].split()[:-1]),
                    'context_length': context_length
                })
            
            print(f"✅ Loaded {len(formatted_data)} LAMBADA examples")
            return formatted_data
            
        except Exception as e:
            print(f"❌ Failed to load LAMBADA dataset: {e}")
            return []
    
    def create_mixed_dataset(self, 
                           msc_examples: int = 1000,
                           narrativeqa_examples: int = 1000, 
                           booksum_examples: int = 500,
                           lambada_examples: int = 500) -> List[Dict[str, Any]]:
        """Create a mixed dataset testing all MARU capabilities."""
        print("🔄 Creating mixed MARU dataset...")
        
        mixed_data = []
        
        # Load each dataset
        mixed_data.extend(self.load_msc_dataset()[:msc_examples])
        mixed_data.extend(self.load_narrativeqa_dataset(max_examples=narrativeqa_examples))
        mixed_data.extend(self.load_booksum_dataset(max_examples=booksum_examples))
        mixed_data.extend(self.load_lambada_dataset()[:lambada_examples])
        
        print(f"🎯 Created mixed dataset with {len(mixed_data)} total examples")
        print(f"   📊 Task distribution:")
        
        task_counts = {}
        for item in mixed_data:
            task_type = item['task_type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        for task, count in task_counts.items():
            print(f"   - {task}: {count} examples")
        
        return mixed_data
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """Save formatted dataset to JSONL file."""
        output_path = self.cache_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"💾 Saved dataset to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = MARUDatasetLoader()
    
    # Create a mixed dataset for MARU training
    mixed_dataset = loader.create_mixed_dataset(
        msc_examples=100,      # Multi-session memory
        narrativeqa_examples=100,  # Long document QA  
        booksum_examples=50,   # Book summarization
        lambada_examples=50    # Long-range dependencies
    )
    
    # Save the dataset
    loader.save_dataset(mixed_dataset, "maru_mixed_dataset.jsonl")
