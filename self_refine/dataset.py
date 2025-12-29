# dataset.py
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import ast
import re
import json


class DatasetHandler(ABC):
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def parse_answer(self, response: str) -> Any:
        pass
    
    @abstractmethod
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        pass
    
    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        for key in ["answer", "ground_truth", "label", "target", "solution"]:
            if key in example:
                return example[key]
        return None


class GraphHandler(DatasetHandler):
    def format_question(self, example: Dict[str, Any]) -> str:
        edges = example.get("edges", [])
        num_nodes = example.get("N") or example.get("num_nodes")
        num_paths = example.get("P") or example.get("num_paths") or 1
        
        if num_nodes is None and edges:
            max_node = max(max(edge[0], edge[1]) for edge in edges if len(edge) >= 2)
            num_nodes = max_node + 1
        
        target_node = num_nodes - 1
        
        edges_str = "\n".join([f"{src} -> {dst}, weight: {weight}" for src, dst, weight in edges])
        
        if num_paths == 1:
            task = f"Find the single shortest path from node 0 to node {target_node}."
        else:
            task = f"Find the top {num_paths} shortest paths from node 0 to node {target_node}."
        
        prompt = f"""You are given a directed graph with {num_nodes} nodes (numbered 0 to {num_nodes-1}) and the following edges:

        Edges (source -> target, weight):
        {edges_str}

        {task}"""
        
        return prompt
    
    def parse_answer(self, response: str) -> Any:
        try:
            text = response.strip()
            
            if "assistant" in text:
                parts = re.split(r'assistant\s*\n', text)
                text = parts[-1] if len(parts) > 1 else text
            
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            
            if text.count('{') > text.count('}'):
                text += '}' * (text.count('{') - text.count('}'))
            if text.count('[') > text.count(']'):
                text += ']' * (text.count('[') - text.count(']'))
            
            text = re.sub(r'\)\}$', ']}', text)
            
            result = json.loads(text)
            
            if not isinstance(result, dict) or "paths" not in result:
                return None
            
            paths_data = result["paths"]
            if not isinstance(paths_data, list) or len(paths_data) == 0:
                return None
            
            parsed_paths = []
            parsed_weights = []
            
            for item in paths_data:
                if not isinstance(item, dict):
                    return None
                
                if "path" not in item or "weight" not in item:
                    return None
                
                path = item["path"]
                weight = item["weight"]
                
                try:
                    path = [int(node) for node in path]
                    weight = int(weight)
                except (ValueError, TypeError):
                    return None
                
                parsed_paths.append(path)
                parsed_weights.append(weight)
            
            return {
                "paths": parsed_paths,
                "weights": parsed_weights
            }
            
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return None
        
    
    def verify_answer(self, parsed_answer: Any, ground_truth: Any) -> bool:
        if parsed_answer is None:
            return False

        pred_paths = parsed_answer.get("paths", [])
        pred_weights = parsed_answer.get("weights", [])
        
        if not pred_paths or not pred_weights:
            return False
        
        pred_path_info_list = []
        for path, weight in zip(pred_paths, pred_weights):
            pred_path_info_list.append({"path": path, "weight": weight})
        
        if "solution" in ground_truth:
            gt_paths = ground_truth["solution"].get("paths", [])
        elif "paths" in ground_truth:
            gt_paths = ground_truth["paths"]
        else:
            return False
        
        if not gt_paths:
            return False
        
        for pred_path_info in pred_path_info_list:
            for gt_path_info in gt_paths:
                if (pred_path_info["path"] == gt_path_info["path"] and 
                    pred_path_info["weight"] == gt_path_info["weight"]):
                    return True
        
        return False


class MMLUMedHandler(DatasetHandler):
    
    def format_question(self, example: Dict[str, Any]) -> str:
        parts = []
        
        question = example.get("question", "")
        if question:
            parts.append(question)
        
        choices = example.get("choices", example.get("options", []))
        if choices:
            choice_lines = []
            for i, choice in enumerate(choices):
                letter = chr(ord('A') + i)
                choice_lines.append(f"{letter}. {choice}")
            parts.append("\n".join(choice_lines))
        
        return "\n\n".join(parts)
    
    def parse_answer(self, response: str) -> Any:
        if not response:
            return None
        
        text = response.strip()
        
        match = re.search(r'(?i)answer\s*[:\-]\s*([A-D])', text)
        if match:
            return match.group(1).upper()
        
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1).upper()
        
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) == 1 and line.upper() in 'ABCD':
                return line.upper()
        
        return None
    
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        if predicted is None or ground_truth is None:
            return False
        
        def normalize(val):
            if isinstance(val, str):
                val = val.strip().upper()
                match = re.search(r'[A-D]', val)
                if match:
                    return match.group(0)
            elif isinstance(val, int):
                if 0 <= val < 4:
                    return chr(ord('A') + val)
            return str(val).upper()
        
        return normalize(predicted) == normalize(ground_truth)
