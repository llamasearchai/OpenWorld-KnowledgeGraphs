"""Advanced knowledge graph extraction methods."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from .store import Triple

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of knowledge extraction."""
    triples: List[Triple]
    entities: Set[str]
    relations: Set[str]
    confidence_scores: Optional[Dict[str, float]] = None


class BaseExtractor(ABC):
    """Base class for knowledge extractors."""
    
    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract knowledge from text."""
        pass


class RegexExtractor(BaseExtractor):
    """Regular expression-based knowledge extractor."""
    
    def __init__(self, patterns: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize with custom patterns.
        
        Args:
            patterns: List of (pattern, relation) tuples
        """
        self.patterns = patterns or self._get_default_patterns()
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), relation) 
            for pattern, relation in self.patterns
        ]
    
    def _get_default_patterns(self) -> List[Tuple[str, str]]:
        """Get default extraction patterns."""
        return [
            # Entity relationships
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(builds|creates|makes)\s+([A-Z][A-Za-z0-9_-]+)\b', 'builds'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(links|connects|joins)\s+([A-Z][A-Za-z0-9_-]+)\b', 'links'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(improves|enhances|optimizes)\s+([A-Z][A-Za-z0-9_-]+)\b', 'improves'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(supports|enables|facilitates)\s+([A-Z][A-Za-z0-9_-]+)\b', 'supports'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(uses|utilizes|employs)\s+([A-Z][A-Za-z0-9_-]+)\b', 'uses'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(requires|needs|depends_on)\s+([A-Z][A-Za-z0-9_-]+)\b', 'requires'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(contains|includes|has)\s+([A-Z][A-Za-z0-9_-]+)\b', 'contains'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(processes|handles|manages)\s+([A-Z][A-Za-z0-9_-]+)\b', 'processes'),
            
            # "is a" relationships
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+is\s+a(?:n)?\s+([A-Z][A-Za-z0-9_-]+)\b', 'is_a'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+are\s+([A-Z][A-Za-z0-9_-]+)\b', 'is_a'),
            
            # "part of" relationships
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+(?:is\s+)?part\s+of\s+([A-Z][A-Za-z0-9_-]+)\b', 'part_of'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+belongs\s+to\s+([A-Z][A-Za-z0-9_-]+)\b', 'belongs_to'),
            
            # Technical relationships
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+implements\s+([A-Z][A-Za-z0-9_-]+)\b', 'implements'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+extends\s+([A-Z][A-Za-z0-9_-]+)\b', 'extends'),
            (r'\b([A-Z][A-Za-z0-9_-]+)\s+inherits\s+from\s+([A-Z][A-Za-z0-9_-]+)\b', 'inherits'),
        ]
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract triples from text using regex patterns."""
        triples = []
        entities = set()
        relations = set()
        
        for pattern, default_relation in self.compiled_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 3:
                    # Pattern with explicit relation
                    subject, relation, obj = groups[0], groups[1], groups[2]
                    relation = relation.lower().replace(' ', '_')
                elif len(groups) == 2:
                    # Pattern with default relation
                    subject, obj = groups[0], groups[1]
                    relation = default_relation
                else:
                    continue
                
                # Clean up entities
                subject = subject.strip()
                obj = obj.strip()
                
                if subject and obj and subject != obj:
                    triple = Triple(s=subject, p=relation, o=obj)
                    triples.append(triple)
                    entities.update([subject, obj])
                    relations.add(relation)
        
        logger.debug(f"Extracted {len(triples)} triples, {len(entities)} entities, {len(relations)} relations")
        
        return ExtractionResult(
            triples=triples,
            entities=entities,
            relations=relations
        )


class NamedEntityExtractor(BaseExtractor):
    """Extract entities and relationships using named entity recognition."""
    
    def __init__(self, min_entity_length: int = 2):
        self.min_entity_length = min_entity_length
        
        # Common entity types to look for
        self.entity_patterns = [
            # Technology terms
            r'\b[A-Z]{2,}(?:[A-Z][a-z]+)*\b',  # Acronyms like RAG, API, HTTP
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase like JavaScript, TensorFlow
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Title Case like Machine Learning
            
            # System components
            r'\b(?:database|server|client|service|system|framework|library|tool|application)\b',
            
            # File and data types
            r'\b\w+\.(?:json|xml|csv|txt|pdf|html|yaml|toml)\b',
            r'\b(?:JSON|XML|CSV|HTML|YAML|TOML)\b',
        ]
        
        self.compiled_entity_patterns = [re.compile(p, re.IGNORECASE) for p in self.entity_patterns]
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and infer relationships."""
        entities = set()
        
        # Extract entities using patterns
        for pattern in self.compiled_entity_patterns:
            for match in pattern.finditer(text):
                entity = match.group().strip()
                if len(entity) >= self.min_entity_length:
                    entities.add(entity)
        
        # Simple co-occurrence based relationship inference
        triples = []
        relations = set()
        
        # If entities appear in the same sentence, infer relationships
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_entities = [e for e in entities if e.lower() in sentence.lower()]
            
            # Create relationships between entities in the same sentence
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    if entity1 != entity2:
                        # Infer relationship type based on context
                        relation = self._infer_relation(sentence, entity1, entity2)
                        if relation:
                            triple = Triple(s=entity1, p=relation, o=entity2)
                            triples.append(triple)
                            relations.add(relation)
        
        return ExtractionResult(
            triples=triples,
            entities=entities,
            relations=relations
        )
    
    def _infer_relation(self, sentence: str, entity1: str, entity2: str) -> Optional[str]:
        """Infer relationship type from sentence context."""
        sentence_lower = sentence.lower()
        
        # Simple heuristics for relationship inference
        if any(word in sentence_lower for word in ['use', 'uses', 'using']):
            return 'uses'
        elif any(word in sentence_lower for word in ['create', 'creates', 'build', 'builds']):
            return 'creates'
        elif any(word in sentence_lower for word in ['contain', 'contains', 'include', 'includes']):
            return 'contains'
        elif any(word in sentence_lower for word in ['connect', 'connects', 'link', 'links']):
            return 'connects'
        elif any(word in sentence_lower for word in ['process', 'processes', 'handle', 'handles']):
            return 'processes'
        else:
            return 'related_to'  # Default relationship


class HybridExtractor(BaseExtractor):
    """Combines multiple extraction methods."""
    
    def __init__(self, extractors: Optional[List[BaseExtractor]] = None):
        """
        Initialize with list of extractors.
        
        Args:
            extractors: List of extractor instances to combine
        """
        self.extractors = extractors or [
            RegexExtractor(),
            NamedEntityExtractor()
        ]
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract using all configured extractors and merge results."""
        all_triples = []
        all_entities = set()
        all_relations = set()
        confidence_scores = {}
        
        for extractor in self.extractors:
            try:
                result = extractor.extract(text)
                
                all_triples.extend(result.triples)
                all_entities.update(result.entities)
                all_relations.update(result.relations)
                
                # Track which extractor found each triple (for confidence scoring)
                for triple in result.triples:
                    triple_key = f"{triple.s}|{triple.p}|{triple.o}"
                    confidence_scores[triple_key] = confidence_scores.get(triple_key, 0) + 0.5
                    
            except Exception as e:
                logger.warning(f"Extractor {extractor.__class__.__name__} failed: {e}")
                continue
        
        # Remove duplicate triples
        unique_triples = []
        seen_triples = set()
        
        for triple in all_triples:
            triple_key = (triple.s, triple.p, triple.o)
            if triple_key not in seen_triples:
                unique_triples.append(triple)
                seen_triples.add(triple_key)
        
        logger.info(f"Combined extraction: {len(unique_triples)} unique triples from {len(all_triples)} total")
        
        return ExtractionResult(
            triples=unique_triples,
            entities=all_entities,
            relations=all_relations,
            confidence_scores=confidence_scores
        )


def extract_knowledge(
    text: str,
    extractor_type: str = "hybrid",
    **extractor_kwargs
) -> ExtractionResult:
    """
    Extract knowledge from text using specified extractor.
    
    Args:
        text: Text to extract knowledge from
        extractor_type: Type of extractor ('regex', 'ner', 'hybrid')
        **extractor_kwargs: Additional arguments for extractor
        
    Returns:
        ExtractionResult with extracted knowledge
    """
    if extractor_type == "regex":
        extractor = RegexExtractor(**extractor_kwargs)
    elif extractor_type == "ner":
        extractor = NamedEntityExtractor(**extractor_kwargs)
    elif extractor_type == "hybrid":
        extractor = HybridExtractor(**extractor_kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    return extractor.extract(text)