"""Advanced knowledge graph querying capabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

from .store import KGStore, Triple

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a knowledge graph query."""
    triples: List[Triple]
    entities: Set[str]
    relations: Set[str]
    paths: Optional[List[List[Triple]]] = None
    scores: Optional[Dict[str, float]] = None


class KGQuerier:
    """Advanced knowledge graph querier."""
    
    def __init__(self, kg_store: KGStore):
        self.kg = kg_store
    
    def find_neighbors(
        self, 
        node: str, 
        max_depth: int = 1, 
        relation_filter: Optional[Set[str]] = None,
        k: int = 10
    ) -> QueryResult:
        """
        Find neighbors of a node with optional filtering.
        
        Args:
            node: Starting node
            max_depth: Maximum depth to search (1 = direct neighbors)
            relation_filter: Set of relations to include (None = all)
            k: Maximum number of results
            
        Returns:
            QueryResult with neighbors
        """
        visited = set()
        all_triples = []
        all_entities = {node}
        all_relations = set()
        
        current_nodes = {node}
        
        for depth in range(max_depth):
            next_nodes = set()
            
            for current_node in current_nodes:
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                # Get outgoing edges
                neighbors = self.kg.neighbors(current_node, k=k)
                
                for triple in neighbors:
                    # Apply relation filter
                    if relation_filter and triple.p not in relation_filter:
                        continue
                    
                    all_triples.append(triple)
                    all_entities.update([triple.s, triple.o])
                    all_relations.add(triple.p)
                    next_nodes.add(triple.o)
                
                # Also get incoming edges
                incoming = self._get_incoming_edges(current_node, k)
                for triple in incoming:
                    if relation_filter and triple.p not in relation_filter:
                        continue
                    
                    all_triples.append(triple)
                    all_entities.update([triple.s, triple.o])
                    all_relations.add(triple.p)
                    next_nodes.add(triple.s)
            
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        return QueryResult(
            triples=all_triples[:k],
            entities=all_entities,
            relations=all_relations
        )
    
    def find_paths(
        self, 
        start: str, 
        end: str, 
        max_path_length: int = 3,
        max_paths: int = 5
    ) -> QueryResult:
        """
        Find paths between two nodes.
        
        Args:
            start: Starting node
            end: Target node
            max_path_length: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            QueryResult with paths
        """
        paths = []
        all_triples = []
        all_entities = set()
        all_relations = set()
        
        def dfs(current: str, target: str, path: List[Triple], visited: Set[str], depth: int):
            if depth > max_path_length or len(paths) >= max_paths:
                return
            
            if current == target and path:
                paths.append(path.copy())
                return
            
            if current in visited:
                return
            
            visited.add(current)
            
            # Get neighbors
            neighbors = self.kg.neighbors(current, k=20)  # Get more for path finding
            
            for triple in neighbors:
                if triple.o not in visited:
                    path.append(triple)
                    dfs(triple.o, target, path, visited.copy(), depth + 1)
                    path.pop()
        
        # Start DFS
        dfs(start, end, [], set(), 0)
        
        # Collect all triples from paths
        for path in paths:
            all_triples.extend(path)
            for triple in path:
                all_entities.update([triple.s, triple.o])
                all_relations.add(triple.p)
        
        return QueryResult(
            triples=all_triples,
            entities=all_entities,
            relations=all_relations,
            paths=paths
        )
    
    def find_by_relation(
        self, 
        relation: str, 
        subject_filter: Optional[str] = None,
        object_filter: Optional[str] = None,
        k: int = 10
    ) -> QueryResult:
        """
        Find triples by relation type.
        
        Args:
            relation: Relation type to search for
            subject_filter: Optional subject filter (partial match)
            object_filter: Optional object filter (partial match)
            k: Maximum number of results
            
        Returns:
            QueryResult with matching triples
        """
        conn = self.kg._conn()
        try:
            # Build query with filters
            query = "SELECT s, p, o FROM edges WHERE p = ?"
            params = [relation]
            
            if subject_filter:
                query += " AND s LIKE ?"
                params.append(f"%{subject_filter}%")
            
            if object_filter:
                query += " AND o LIKE ?"
                params.append(f"%{object_filter}%")
            
            query += " LIMIT ?"
            params.append(k)
            
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
            
            triples = [Triple(s=row[0], p=row[1], o=row[2]) for row in results]
            entities = set()
            relations = {relation}
            
            for triple in triples:
                entities.update([triple.s, triple.o])
            
            return QueryResult(
                triples=triples,
                entities=entities,
                relations=relations
            )
            
        finally:
            conn.close()
    
    def find_similar_entities(
        self, 
        entity: str, 
        similarity_threshold: float = 0.3,
        k: int = 10
    ) -> QueryResult:
        """
        Find entities similar to the given entity.
        
        Args:
            entity: Entity to find similar entities for
            similarity_threshold: Minimum similarity score
            k: Maximum number of results
            
        Returns:
            QueryResult with similar entities
        """
        # Simple similarity based on shared neighbors
        entity_neighbors = self.kg.neighbors(entity, k=50)
        entity_relations = {triple.p for triple in entity_neighbors}
        entity_targets = {triple.o for triple in entity_neighbors}
        
        # Get all entities
        conn = self.kg._conn()
        try:
            cursor = conn.execute("SELECT DISTINCT s FROM edges UNION SELECT DISTINCT o FROM edges")
            all_entities = [row[0] for row in cursor.fetchall() if row[0] != entity]
            
            similarities = []
            
            for other_entity in all_entities[:100]:  # Limit for performance
                other_neighbors = self.kg.neighbors(other_entity, k=50)
                other_relations = {triple.p for triple in other_neighbors}
                other_targets = {triple.o for triple in other_neighbors}
                
                # Calculate Jaccard similarity
                relation_similarity = self._jaccard_similarity(entity_relations, other_relations)
                target_similarity = self._jaccard_similarity(entity_targets, other_targets)
                
                overall_similarity = (relation_similarity + target_similarity) / 2
                
                if overall_similarity >= similarity_threshold:
                    similarities.append((other_entity, overall_similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_entities = similarities[:k]
            
            # Create triples showing similarity relationships
            triples = []
            entities = {entity}
            relations = {'similar_to'}
            scores = {}
            
            for similar_entity, score in similar_entities:
                triple = Triple(s=entity, p='similar_to', o=similar_entity)
                triples.append(triple)
                entities.add(similar_entity)
                scores[f"{entity}|similar_to|{similar_entity}"] = score
            
            return QueryResult(
                triples=triples,
                entities=entities,
                relations=relations,
                scores=scores
            )
            
        finally:
            conn.close()
    
    def get_entity_stats(self, entity: Optional[str] = None) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Get statistics about entities and relations.
        
        Args:
            entity: Optional specific entity to get stats for
            
        Returns:
            Dictionary with statistics
        """
        conn = self.kg._conn()
        try:
            if entity:
                # Stats for specific entity
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE s = ? OR o = ?", 
                    (entity, entity)
                )
                total_connections = cursor.fetchone()[0]
                
                cursor = conn.execute(
                    "SELECT p, COUNT(*) FROM edges WHERE s = ? GROUP BY p", 
                    (entity,)
                )
                outgoing_relations = dict(cursor.fetchall())
                
                cursor = conn.execute(
                    "SELECT p, COUNT(*) FROM edges WHERE o = ? GROUP BY p", 
                    (entity,)
                )
                incoming_relations = dict(cursor.fetchall())
                
                return {
                    'entity': entity,
                    'total_connections': total_connections,
                    'outgoing_relations': outgoing_relations,
                    'incoming_relations': incoming_relations
                }
            else:
                # Global stats
                cursor = conn.execute("SELECT COUNT(*) FROM edges")
                total_triples = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT s) FROM edges")
                total_subjects = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT o) FROM edges")
                total_objects = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT p) FROM edges")
                total_relations = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT p, COUNT(*) FROM edges GROUP BY p ORDER BY COUNT(*) DESC")
                relation_counts = dict(cursor.fetchall())
                
                total_entities = len(set([entity] + 
                    [row[0] for row in conn.execute("SELECT DISTINCT s FROM edges").fetchall()] +
                    [row[0] for row in conn.execute("SELECT DISTINCT o FROM edges").fetchall()]
                ))
                
                return {
                    'total_triples': total_triples,
                    'total_entities': total_entities,
                    'total_subjects': total_subjects,
                    'total_objects': total_objects,
                    'total_relations': total_relations,
                    'relation_counts': relation_counts
                }
                
        finally:
            conn.close()
    
    def _get_incoming_edges(self, node: str, k: int = 10) -> List[Triple]:
        """Get incoming edges for a node."""
        conn = self.kg._conn()
        try:
            cursor = conn.execute(
                "SELECT s, p, o FROM edges WHERE o = ? LIMIT ?", 
                (node, k)
            )
            return [Triple(s=row[0], p=row[1], o=row[2]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0