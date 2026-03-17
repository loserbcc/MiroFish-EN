"""
Graphiti-core Monkey Patch

Workaround for graphiti-core Issue #683:
LLM-generated nested attributes can cause Neo4j writes to fail
(Neo4j property values only accept primitive types or arrays thereof)

Patch strategy:
- Intercept bulk_utils.add_nodes_and_edges_bulk_tx
- Convert nested dict/list to JSON strings before writing to Neo4j
"""

import json
import functools
from typing import Any, Dict

from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_patch')

_patch_applied = False


def sanitize_for_neo4j(value: Any, path: str = "") -> Any:
    """
    Recursively sanitize values to comply with Neo4j property constraints

    Neo4j only accepts:
    - Primitive types: str, int, float, bool, None
    - Arrays of primitive types (no nesting)

    Strategy:
    - Nested dict -> JSON string
    - Nested list (containing dict) -> JSON string
    - Simple list (only primitive types) -> keep as-is
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        # dict needs to be serialized to JSON string
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cannot serialize dict property {path}: {e}")
            return str(value)

    if isinstance(value, (list, tuple)):
        # Check if it's a simple array (only primitive types)
        is_simple = all(isinstance(v, (str, int, float, bool, type(None))) for v in value)
        if is_simple:
            return list(value)
        # Contains complex types, serialize to JSON
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cannot serialize list property {path}: {e}")
            return str(value)

    # Other types converted to string
    return str(value)


def sanitize_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize an entire attributes dictionary
    """
    if not attrs:
        return {}

    sanitized = {}
    for key, value in attrs.items():
        sanitized[key] = sanitize_for_neo4j(value, path=key)
    return sanitized


def apply_patch() -> bool:
    """
    Apply the monkey-patch to graphiti-core

    Returns:
        bool: Whether the patch was successfully applied
    """
    global _patch_applied

    if _patch_applied:
        logger.debug("Graphiti patch already applied, skipping")
        return True

    try:
        from graphiti_core.utils import bulk_utils

        # Save the original function
        original_add_nodes_and_edges_bulk_tx = bulk_utils.add_nodes_and_edges_bulk_tx

        @functools.wraps(original_add_nodes_and_edges_bulk_tx)
        async def patched_add_nodes_and_edges_bulk_tx(
            tx,  # GraphDriverSession (from session.execute_write)
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver,
        ):
            """
            Patched version: sanitize node/edge attributes before Neo4j write

            Signature matches graphiti-core 0.25.0's add_nodes_and_edges_bulk_tx:
            (tx, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder, driver)
            """
            # Sanitize entity_nodes attributes
            for node in entity_nodes:
                if hasattr(node, 'attributes') and node.attributes:
                    node.attributes = sanitize_attributes(node.attributes)

            # Sanitize entity_edges attributes
            for edge in entity_edges:
                if hasattr(edge, 'attributes') and edge.attributes:
                    edge.attributes = sanitize_attributes(edge.attributes)

            # Call the original function
            return await original_add_nodes_and_edges_bulk_tx(
                tx,
                episodic_nodes,
                episodic_edges,
                entity_nodes,
                entity_edges,
                embedder,
                driver,
            )

        # Apply the patch
        bulk_utils.add_nodes_and_edges_bulk_tx = patched_add_nodes_and_edges_bulk_tx

        _patch_applied = True
        logger.info("Graphiti bulk_utils patch applied successfully")
        return True

    except ImportError as e:
        logger.warning(f"Cannot import graphiti_core.utils.bulk_utils: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply Graphiti patch: {e}")
        return False
