from neo4j import GraphDatabase
import networkx as nx

uri = ""
username = ""
password = ""
BATCH_SIZE = 1000

def create_nodes(tx, nodes):
    """
    Creates or updates nodes in the Neo4j database.

    This function takes a list of nodes, and for each node, it either creates a new node or updates an existing one based on the node's `id`. The attributes of each node are set or updated as specified in the `attributes` field.

    Args:
        tx (neo4j.Transaction): The active Neo4j transaction.
        nodes (list): A list of dictionaries, where each dictionary represents a node with the following structure:
            - 'id': The unique identifier for the node.
            - 'attributes': A dictionary of key-value pairs representing node attributes.
    """    
    tx.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        SET n += node.attributes
        """,
        nodes=nodes
    )

def create_relationships(tx, relationships):
    """
    Creates or updates relationships between nodes in the Neo4j database.

    This function takes a list of relationships and for each relationship, it either creates a new relationship or updates an existing one
    based on the source and target node identifiers (`source_id` and `target_id`). The relationship's attributes are set or updated
    as specified in the `attributes` field.

    Args:
        tx (neo4j.Transaction): The active Neo4j transaction.
        relationships (list): A list of dictionaries, where each dictionary represents a relationship with the following structure:
            - 'source_id': The unique identifier of the source node.
            - 'target_id': The unique identifier of the target node.
            - 'attributes': A dictionary of key-value pairs representing relationship attributes.
    """    
    tx.run(
        """
        UNWIND $relationships AS rel
        MATCH (a:Node {id: rel.source_id})
        MATCH (b:Node {id: rel.target_id})
        MERGE (a)-[r:RELATES_TO]-(b)
        SET r += rel.attributes
        """,
        relationships=relationships
    )

def create_index(tx):
    """
    Creates a unique constraint on the `id` property of the `Node` label in the Neo4j database.

    This function ensures that the `id` property of each `Node` is unique, preventing the creation of nodes
    with duplicate `id` values. It is typically used for enforcing data integrity and speeding up lookup operations.

    Args:
        tx (neo4j.Transaction): The active Neo4j transaction.
    """    
    tx.run("CREATE CONSTRAINT FOR (n:Node) REQUIRE n.id IS UNIQUE")
