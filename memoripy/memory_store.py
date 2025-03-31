# memory_store.py
# Apache 2.0 license, Modified by Awilen Bernkastel

from itertools import combinations
import logging
import math
import numpy as np
import time
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict
from .interaction import Interaction

logger = logging .getLogger("memoripy")

class MemoryStore:
    def __init__(self):
        self.short_term_memory = []  # Short-term memory interactions
        self.long_term_memory = []   # Long-term memory interactions
        self.decayed_memory = []     # Short-term memory ready for forgetting
        self.graph = nx.Graph()      # Graph for bidirectional associations
        self.semantic_memory = defaultdict(list)  # Semantic memory clusters
        self.kmeans:KMeans = None

    def add_interaction(self, interaction: Interaction):
        # Reshape the embedding if necessary
        interaction.embedding = np.array(interaction.embedding).reshape(1, -1)

        # Save the interaction data to short-term memory
        self.short_term_memory.append(interaction)

        # Update graph with bidirectional associations
        self.update_graph(interaction.concepts)

        # Update clusters with new interaction
        self.cluster_interactions()

        logger.info(f"Total interactions stored in short-term memory: {len(self.short_term_memory)}")

    def initialize_memory(self, interactions: list[Interaction]):
        self.short_term_memory = []
        self.long_term_memory = []
        # Add all interactions at once
        logger.info(f"Adding {len(interactions)} interactions to short-term memory...")
        for interaction in interactions:
            self.short_term_memory.append(interaction)
            self.update_graph(interaction.concepts)
        # Only do the clustering once
        self.cluster_interactions()

    def update_graph(self, concepts):
        # Add edges between concepts (associations) (nodes are added to the graph if they don't exist)
        # The graph is already bidirectional, we don't need to update (concept1, concept2) and (concept2, concept1),
        # so combinations is better than permutations.
        for concept1, concept2 in combinations(concepts, 2):
            self.graph.add_edge(concept1, concept2, weight=self.graph.get_edge_data(concept1, concept2, {'weight': 0})['weight'] + 1)

    def cleanup_concepts(self):
        # Reduce graph weights for interactions about to be forgotten
        for im in self.decayed_memories:
            for concept1, concept2 in combinations(im.concepts, 2):
                self.graph[concept1][concept2]['weight'] -= 1

        # Filter the concepts that can be removed from the interactions ready to be forgotten, off of the remaining interactions
        concepts_potentially_to_remove = {x.concepts for x in self.decayed_memory}
        concepts_remaining = {x.concepts for x in self.short_term_memory}
        concepts_to_remove = filter(lambda x: x not in concepts_remaining, concepts_potentially_to_remove)

        # Remove the necessary nodes from the concept graph if there's no interaction containing the concept anymore
        self.graph.remove_nodes_from(concepts_to_remove)

    def retrieve(self, query_interaction: Interaction, similarity_threshold=0.4, exclude_last_n=0):
        if len(self.short_term_memory) == 0:
            logger.info("No interactions available in short-term memory for retrieval.")
            return []

        logger.info("Retrieving relevant interactions from short-term memory...")
        relevant_interactions = []
        current_time = time.time()

        # Compute adjusted similarity for each interaction
        if len(self.short_term_memory) > exclude_last_n:
            # Extract by adjusted similarity
            relevant_interactions = list( # Cast to a list
                                        filter(
                                            lambda x,_ : x >= similarity_threshold, # Keep entries above the similarity threshold
                                            # Build the list of (adjusted_similarity, interaction) tuples for each interaction
                                            [(x.adjusted_similarity(query_interaction, current_time), x) for x in self.short_term_memory[:-exclude_last_n]]
                                        )
                                    )

        # Spreading activation
        activated_concepts = self.spreading_activation(query_interaction)

        # Integrate spreading activation scores
        final_interactions = []
        for score, interaction in relevant_interactions:
            activation_score = sum([activated_concepts.get(c, 0) for c in interaction.concepts])
            total_score = score + activation_score
            final_interactions.append((total_score, interaction))

        # Sort interactions based on total_score
        final_interactions = sorted(final_interactions, key=lambda x: x[0], reverse=True)
        final_interactions = [interaction for _, interaction in final_interactions]

        # Retrieve from semantic memory
        semantic_interactions = self.retrieve_from_semantic_memory(query_interaction)
        final_interactions.extend(semantic_interactions)
        final_interactions = list(set(final_interactions))

        # Update interaction access
        self.update_interactions(final_interactions, current_time, exclude_last_n)

        if self.decayed_memory:
            # Filter the interactions marked for deletion
            self.short_term_memory = list(filter(lambda x: x not in self.decayed_memory, self.short_term_memory))
            # Cleanup the concepts of the decayed memories
            self.cleanup_concepts()
            # Make sure decayed memories do not end up in the final_interactions from the semantic memory retrieval
            final_interactions = list(filter(lambda x: x not in self.decayed_memory, final_interactions))
            # Recluster interactions
            # Not necessarily needed as the next interaction added will recluster them anyway,
            # enable this only to ensure data consistency.
            # self.cluster_interactions()

        # Sort retrieved interaction by timestamp
        final_interactions = sorted(final_interactions, key = lambda x: x.timestamp)

        logger.info(f"Retrieved {len(final_interactions)} relevant interactions from memory.")
        return final_interactions

    def update_interactions(self, relevant_interactions, current_time, exclude_last_n):
        self.decayed_memory = []
        # Update access count and timestamp for relevant interactions
        if len(self.short_term_memory) > exclude_last_n:
            ri = [r[1] for r in relevant_interactions]
            for im in self.short_term_memory[:-exclude_last_n]:
                if im in ri:
                    # Increase access count and decay factor for relevant interaction
                    im.access_count += 1
                    im.decay_factor *= 1.1  # Increase by 10%, adjust as needed
                    im.last_accessed = current_time
                    logger.debug(f"Increased decay factor for interaction {im.id}: {im.decay_factor:0.3f}")

                    if im not in self.long_term_memory and im.access_count > 10:
                        self.long_term_memory.append(im)
                        logger.info(f"Moved interaction {im.id} to long-term memory.")

                else:
                    # Apply decay for non-relevant interactions
                    im.decay_factor /= 1.1
                    logger.debug(f"Decreased decay factor for interaction {im.id}: {im.decay_factor:0.3f}")
                    # Mark non-committed short-term memories for forgetting
                    if im not in self.long_term_memory and im.decay_factor < 0.3: # From 1, that's about 12 interactions without reinforcement
                        self.decayed_memory.append(im)
                        logger.info(f"Moved interaction {im.id} to decayed memory.")
                    # Allow very heavily decayed interactions to be forgotten, even if in long-term memory 
                    elif im in self.long_term_memory and im.decay_factor < 0.000000000126: # From 2.56 (10 rememberances from 1.0), that's about 250 interactions without reinforcement
                        self.decayed_memory.append(im)
                        self.long_term_memory.pop(im)
                        logger.info(f"Moved interaction {im.id} from long-term memory to decayed memory.")

    def spreading_activation(self, query_interaction: Interaction):
        logger.info("Spreading activation for concept associations...")
        activated_nodes = {}
        initial_activation = 1.0
        decay_factor = 0.5  # How much the activation decays each step

        # Initialize activation levels
        for concept in query_interaction.concepts:
            activated_nodes[concept] = initial_activation

        # Spread activation over the graph
        for step in range(2):  # Number of steps to spread activation
            new_activated_nodes = {}
            for node in activated_nodes:
                if node in self.graph:  # Check if the node exists in the graph
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in activated_nodes:
                            weight = self.graph[node][neighbor]['weight']
                            new_activation = activated_nodes[node] * decay_factor * weight
                            new_activated_nodes[neighbor] = new_activated_nodes.get(neighbor, 0) + new_activation
            activated_nodes.update(new_activated_nodes)

        logger.info(f"Concepts activated after spreading: {activated_nodes}")
        return activated_nodes

    def cluster_interactions(self):

        logger.info(f"Clustering {len(self.short_term_memory)} interactions to create hierarchical memory...")
        if len(self.short_term_memory) < 2:
            logger.info("Not enough interactions to perform clustering.")
            return

        num_clusters = math.floor(math.sqrt(len(self.short_term_memory))) # Allow a growing number of clusters without going overboard.

        # If the number of clusters needs to change due to expanding or shrinking of self.short_term_memory
        if num_clusters != len(self.semantic_memory.keys()):
            # Recluster interactions
            logging.info("Clustering all interactions...")
            self.semantic_memory = defaultdict(list)
            embeddings_matrix = np.vstack([im.embedding for im in self.short_term_memory])
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_matrix)
            # Build semantic memory clusters
            interactions = list(zip(self.kmeans.labels_, self.short_term_memory))
            labeled_interactions = sorted(interactions, key=lambda x: x[0])
            for label, interaction in labeled_interactions:
                self.semantic_memory[label].append(interaction)
        else:
            # Cluster the last interaction within the existing clusters, avoids recomputing the clusters for each interaction
            # This leads to a degradation of the quality of clusters until the interactions are clustered again.
            logging.info("Addint new interaction to clusters...")
            last_interaction = self.short_term_memory[-1]
            labels = self.kmeans.predict(last_interaction.embedding)
            self.semantic_memory[labels[0]].append(last_interaction)

        logger.info(f"Clustering completed. Total clusters formed: {num_clusters}")

    def retrieve_from_semantic_memory(self, query_interaction: Interaction):
        logger.info("Retrieving interactions from semantic memory...")
        # Find the cluster closest to the query
        cluster_similarities = {}
        for label, interactions in self.semantic_memory.items():
            # Calculate centroid of the cluster
            cluster_embeddings = np.vstack([interaction.embedding for interaction in interactions])
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            centroid_norm = normalize(centroid)
            similarity = query_interaction.embedding_similarity(centroid_norm)
            cluster_similarities[label] = similarity

        # Select the most similar cluster
        if not cluster_similarities:
            return []
        best_cluster_label = max(cluster_similarities, key=cluster_similarities.get)
        logger.info(f"Best matching cluster identified: {best_cluster_label}")

        # Retrieve interactions from the best cluster
        cluster_items = self.semantic_memory[best_cluster_label]

        # Sort interactions based on similarity to the query
        interactions = list(sorted(cluster_items, key=lambda x: query_interaction.embedding_similarity(x.embedding), reverse=True))
        semantic_interactions = interactions[:5]  # Limit to top 5 interactions

        logger.info(f"Retrieved {len(semantic_interactions)} interactions from the best matching cluster.")
        return semantic_interactions
