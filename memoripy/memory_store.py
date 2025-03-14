# memory_store.py
# Apache 2.0 license, Modified by Awilen Bernkastel

import faiss
import logging
import numpy as np
import time
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict
from .interaction_data import InteractionData

class MemoryStore:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.short_term_memory = []  # Short-term memory interactions
        self.long_term_memory = []   # Long-term memory interactions
        self.graph = nx.Graph()      # Graph for bidirectional associations
        self.semantic_memory = defaultdict(list)  # Semantic memory clusters
        self.cluster_labels = []     # Labels for each interaction's cluster

    def add_interaction(self, interaction):
        im = InteractionData()
        im.id = interaction['id']
        im.prompt = interaction['prompt']
        im.output = interaction['output']
        im.embedding = np.array(interaction['embedding']).reshape(1, -1)
        im.timestamp = interaction.get('timestamp', time.time())  # Use current time if 'timestamp' is missing
        im.access_count = interaction.get('access_count', 1)
        im.concepts = set(interaction.get('concepts', []))
        im.decay_factor = interaction.get('decay_factor', 1.0)

        logging.info(f"Adding new interaction to short-term memory: '{interaction['prompt']}'")
        # Save the interaction data to short-term memory
        self.short_term_memory.append(im)

        self.index.add(im.embedding)

        # Update graph with bidirectional associations
        self.update_graph(im.concepts)

        logging.info(f"Total interactions stored in short-term memory: {len(self.short_term_memory)}")

    def update_graph(self, concepts):
        # Use the saved concepts to update the graph
        for concept in concepts:
            self.graph.add_node(concept)
        # Add edges between concepts (associations)
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 != concept2:
                    if self.graph.has_edge(concept1, concept2):
                        self.graph[concept1][concept2]['weight'] += 1
                    else:
                        self.graph.add_edge(concept1, concept2, weight=1)

    def retrieve(self, query_embedding, query_concepts, similarity_threshold=40, exclude_last_n=0):
        if len(self.short_term_memory) == 0:
            logging.info("No interactions available in short-term memory for retrieval.")
            return []

        logging.info("Retrieving relevant interactions from short-term memory...")
        relevant_interactions = []
        current_time = time.time()
        decay_rate = 0.0001  # Adjust decay rate as needed

        # Normalize embeddings for cosine similarity
        normalized_embeddings = [im.normalize_embedding() for im in self.short_term_memory]
        query_embedding_norm = normalize(query_embedding)

        # Track indices of relevant interactions
        relevant_indices = set()

        # Calculate adjusted similarity for each interaction
        for idx, im in enumerate(self.short_term_memory[:-exclude_last_n]):
            # Cosine similarity
            similarity = cosine_similarity(query_embedding_norm, normalized_embeddings[idx])[0][0] * 100
            # Time-based decay
            time_diff = current_time - im.timestamp
            im.decay_factor = im.get('decay_factor', 1.0) * np.exp(-decay_rate * time_diff)
            # Reinforcement
            reinforcement_factor = np.log1p(im.access_count)
            # Adjusted similarity
            adjusted_similarity = similarity * im.decay_factor * reinforcement_factor
            logging.info(f"Interaction {im.id} - Adjusted similarity score: {adjusted_similarity:.2f}%")

            if adjusted_similarity >= similarity_threshold:
                # Mark interaction as relevant
                relevant_indices.add(im.id)
                # Update access count and timestamp for relevant interactions
                im.access_count += 1
                im.timestamp = current_time
                logging.debug(f"Updated access count for interaction {im.id}: {im.access_count}")

                # Move interaction to long-term memory if access count exceeds 10
                if im.access_count > 10 and im not in self.long_term_memory:
                    self.long_term_memory.append(im)
                    logging.info(f"Moved interaction {im.id} to long-term memory.")

                # Increase decay factor for relevant interaction
                im.decay_factor *= 1.1  # Increase by 10% or adjust as needed

                # Add to the list of relevant interactions
                relevant_interactions.append((adjusted_similarity, im))
            else:
                logging.debug(f"Interaction {self.short_term_memory[idx]['id']} was not relevant (similarity: {adjusted_similarity:.2f}%).")

        # Decrease decay factor for non-relevant interactions
        for im in self.short_term_memory:
            if im.id not in relevant_indices:
                # Apply decay for non-relevant interactions
                im.decay_factor *= 0.9  # Decrease by 10% or adjust as needed

        # Spreading activation
        activated_concepts = self.spreading_activation(query_concepts)

        # Integrate spreading activation scores
        final_interactions = []
        for score, interaction in relevant_interactions:
            activation_score = sum([activated_concepts.get(c, 0) for c in interaction.concepts])
            total_score = score + activation_score
            interaction.total_score = total_score
            final_interactions.append((total_score, interaction))

        # Sort interactions based on total_score
        final_interactions.sort(key=lambda x: x[0], reverse=True)
        final_interactions = [interaction for _, interaction in final_interactions]

        # Retrieve from semantic memory
        semantic_interactions = self.retrieve_from_semantic_memory(query_embedding_norm)
        final_interactions.extend(semantic_interactions)

        logging.info(f"Retrieved {len(final_interactions)} relevant interactions from memory.")
        return final_interactions

    def spreading_activation(self, query_concepts):
        logging.info("Spreading activation for concept associations...")
        activated_nodes = {}
        initial_activation = 1.0
        decay_factor = 0.5  # How much the activation decays each step

        # Initialize activation levels
        for concept in query_concepts:
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

        logging.info(f"Concepts activated after spreading: {activated_nodes}")
        return activated_nodes

    def cluster_interactions(self):
        logging.info("Clustering interactions to create hierarchical memory...")
        if len(self.short_term_memory) < 2:
            logging.info("Not enough interactions to perform clustering.")
            return

        embeddings_matrix = np.vstack([im.embedding for im in self.short_term_memory])
        num_clusters = min(10, len(self.short_term_memory))  # Adjust number of clusters based on the number of interactions
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_matrix)
        self.cluster_labels = kmeans.labels_

        # Build semantic memory clusters
        for idx, label in enumerate(self.cluster_labels):
            im = self.short_term_memory[idx]
            self.semantic_memory[label].append((im.embedding, im))

        logging.info(f"Clustering completed. Total clusters formed: {num_clusters}")

    def retrieve_from_semantic_memory(self, query_embedding_norm):
        logging.info("Retrieving interactions from semantic memory...")
        current_time = time.time()
        # Find the cluster closest to the query
        cluster_similarities = {}
        for label, items in self.semantic_memory.items():
            # Calculate centroid of the cluster
            cluster_embeddings = np.vstack([e for e, _ in items])
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            centroid_norm = normalize(centroid)
            similarity = cosine_similarity(query_embedding_norm, centroid_norm)[0][0]
            cluster_similarities[label] = similarity

        # Select the most similar cluster
        if not cluster_similarities:
            return []
        best_cluster_label = max(cluster_similarities, key=cluster_similarities.get)
        logging.info(f"Best matching cluster identified: {best_cluster_label}")

        # Retrieve interactions from the best cluster
        cluster_items = self.semantic_memory[best_cluster_label]
        interactions = [(e, i) for e, i in cluster_items]

        # Sort interactions based on similarity to the query
        interactions.sort(key=lambda x: cosine_similarity(query_embedding_norm, normalize(x[0]))[0][0], reverse=True)
        semantic_interactions = [interaction for _, interaction in interactions[:5]]  # Limit to top 5 interactions

        # Update access count for these retrieved interactions
        for interaction in semantic_interactions:
            interaction_id = interaction.id
            idx = next((i for i, item in enumerate(self.short_term_memory) if item['id'] == interaction_id), None)
            if idx is not None:
                im = self.short_term_memory[idx]
                im.timestamp = current_time
                im.access_count += 1
                logging.debug(f"Updated access count for interaction {interaction_id}: {im.access_count}")

        logging.info(f"Retrieved {len(semantic_interactions)} interactions from the best matching cluster.")
        return semantic_interactions
