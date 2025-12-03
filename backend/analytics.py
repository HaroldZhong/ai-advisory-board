import json
import os
from collections import defaultdict
from typing import Dict, Any, List
from .config import DATA_DIR
from .logger import logger

def get_analytics() -> Dict[str, Any]:
    """
    Calculate aggregate performance statistics for all models.
    """
    stats = defaultdict(lambda: {'total_rank': 0, 'count': 0, 'wins': 0})
    
    if not os.path.exists(DATA_DIR):
        return {"models": []}
        
    processed_files = 0
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.json'):
            continue
            
        try:
            with open(os.path.join(DATA_DIR, filename), 'r') as f:
                data = json.load(f)
                
            for msg in data.get('messages', []):
                # Check for stage2 data (rankings)
                if 'stage2' in msg and msg['stage2']:
                    # We need the mapping from anonymous labels to real model names
                    # This is stored in metadata
                    metadata = msg.get('metadata', {})
                    label_to_model = metadata.get('label_to_model', {})
                    
                    if not label_to_model:
                        continue
                        
                    # Process each judge's ranking
                    for ranking_entry in msg['stage2']:
                        # ranking_entry['parsed_ranking'] is a list of labels ["Response C", "Response A"...]
                        parsed = ranking_entry.get('parsed_ranking', [])
                        
                        if not parsed:
                            continue
                            
                        for rank, label in enumerate(parsed, start=1):
                            model_id = label_to_model.get(label)
                            if model_id:
                                stats[model_id]['total_rank'] += rank
                                stats[model_id]['count'] += 1
                                if rank == 1:
                                    stats[model_id]['wins'] += 1
            
            processed_files += 1
                                    
        except Exception as e:
            logger.error(f"Error processing analytics for {filename}: {e}")
            continue
            
    # Format results
    results = []
    for model_id, data in stats.items():
        avg_rank = data['total_rank'] / data['count'] if data['count'] > 0 else 0
        win_rate = (data['wins'] / data['count']) * 100 if data['count'] > 0 else 0
        results.append({
            "model": model_id,
            "average_rank": round(avg_rank, 2),
            "win_rate": round(win_rate, 1),
            "evaluations": data['count']
        })
        
    # Sort by average rank (lower is better)
    results.sort(key=lambda x: x['average_rank'])
    
    logger.info(f"Generated analytics from {processed_files} conversations")
    return {"models": results}
