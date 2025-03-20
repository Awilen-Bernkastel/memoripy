# memory_store.py
# Apache 2.0 license, Modified by Awilen Bernkastel

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
    def __init__(self):
        self.short_term_memory = []  # Short-term memory interactions
        self.long_term_memory = []   # Long-term memory interactions
        self.decayed_memory = []     # Short-term memory ready for forgetting
        self.graph = nx.Graph()      # Graph for bidirectional associations
        self.semantic_memory = defaultdict(list)  # Semantic memory clusters
        self.cluster_labels = []     # Labels for each interaction's cluster

    def add_interaction(self, interaction: InteractionData):
        # Reshape the embedding if necessary
        interaction.embedding = np.array(interaction.embedding).reshape(1, -1)

        logging.info(f"Adding new interaction to short-term memory: '{interaction['prompt']}'")
        # Save the interaction data to short-term memory
        self.short_term_memory.append(interaction)

        # Update graph with bidirectional associations
        self.update_graph(interaction.concepts)

        # Update clusters with new interaction
        self.cluster_interactions()

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

    def cleanup_concepts(self):
        concepts_potentially_to_remove = set([m.concepts for m in self.decayed_memory])
        concepts_remaining = set([m.concepts for m in self.short_term_memory]).union(set([m.concepts for m in self.long_term_memory]))
        concepts_to_remove = list(filter(lambda x: x in concepts_remaining, concepts_potentially_to_remove))
        for concept in concepts_to_remove:
            # Remove the necessary nodes from the concept graph if there's no interaction containing the concept anymore
            self.graph.remove_node(concept)

    def retrieve(self, query_embedding, query_concepts, similarity_threshold=40, exclude_last_n=0):
        if len(self.short_term_memory) == 0:
            logging.info("No interactions available in short-term memory for retrieval.")
            return []

        logging.info("Retrieving relevant interactions from short-term memory...")
        relevant_interactions = []
        current_time = time.time()
        decay_rate = 0.0001  # Adjust decay rate as needed

        # Normalize embeddings for cosine similarity
        query_embedding_norm = normalize(query_embedding)

        # Calculate adjusted similarity for each interaction
        if len(self.short_term_memory) > exclude_last_n:
            for interaction in self.short_term_memory[:-exclude_last_n]:
                similarity = cosine_similarity(query_embedding_norm, interaction.normalize_embedding())[0][0] * 100
                time_diff = current_time - interaction.last_accessed
                interaction.decay_factor = interaction.get('decay_factor', 1.0) * np.exp(-decay_rate * time_diff)
                reinforcement_factor = np.log1p(interaction.access_count)
                adjusted_similarity = similarity * interaction.decay_factor * reinforcement_factor
                logging.info(f"Interaction {interaction.id} - Adjusted similarity score: {adjusted_similarity:.2f}%")

                if adjusted_similarity >= similarity_threshold:
                    # Add to the list of relevant interactions
                    relevant_interactions.append((adjusted_similarity, interaction))

        # Update interaction access
        self.update_interactions(relevant_interactions, current_time, exclude_last_n)

        if self.decayed_memory:
            self.cleanup_concepts()
            # Filter the interactions marked for deletion
            self.short_term_memory = list(filter(lambda x: x in self.decayed_memory, self.short_term_memory))
            # Recluster interactions
            self.cluster_interactions()

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

    def update_interactions(self, relevant_interactions, current_time, exclude_last_n):
        self.decayed_memory = []
        # Update access count and timestamp for relevant interactions
        if len(self.short_term_memory) > exclude_last_n:
            ri = [r[1] for r in relevant_interactions]
            for im in self.short_term_memory[:-exclude_last_n]:
                if im in ri:
                    im.access_count += 1
                    im.last_accessed = current_time
                    logging.debug(f"Updated access count for interaction {im.id}: {im.access_count}")

                    # Move interaction to long-term memory if access count exceeds 10
                    if im.access_count > 10 and im not in self.long_term_memory:
                        self.long_term_memory.append(im)
                        logging.info(f"Moved interaction {im.id} to long-term memory.")

                        # Increase decay factor for relevant interaction
                        im.decay_factor *= 1.1  # Increase by 10% or adjust as needed
                else:
                    # Apply decay for non-relevant interactions
                    im.decay_factor *= 0.9
                    # Mark for forgetting
                    if im not in self.long_term_memory and im.decay_factor < 0.1:
                        self.decayed_memory.append(im)
        for im in self.decayed_memory:
            self.short_term_memory.pop(im)

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
            self.semantic_memory = {}
            return

        embeddings_matrix = np.vstack([im.embedding for im in self.short_term_memory])
        num_clusters = min(10, len(self.short_term_memory))  # Adjust number of clusters based on the number of interactions
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_matrix)
        self.cluster_labels = kmeans.labels_

        # Build semantic memory clusters
        self.semantic_memory = {label: [im for im in self.short_term_memory if im.label == label] for label in set(self.cluster_labels)}

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
            idx = next((i for i, item in enumerate(self.short_term_memory) if item.id == interaction_id), None)
            if idx is not None:
                im = self.short_term_memory[idx]
                im.last_accessed = current_time
                im.access_count += 1
                logging.debug(f"Updated access count for interaction {interaction_id}: {im.access_count}")

        logging.info(f"Retrieved {len(semantic_interactions)} interactions from the best matching cluster.")
        return semantic_interactions

try:
    import os
    os.environ['USE_FAISS']
except KeyError:
    pass
else:
    try:
        import faiss
    except ImportError:
        print("Error: USE_FAISS is set, but faiss-cpu is not installed. You can install it with 'pip install faiss-cpu'.")
        print("If pip compiles faiss-cpu from source, the source package assumes faiss is already built and installed on the system.")
        print("Read more at https://github.com/kyamagu/faiss-wheels")
        exit(1)

    __memory_store___init__ = MemoryStore.__init__
    def __memory_store_faiss___init__(self, dimension=1536):
        __memory_store___init__(self)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    __memory_store_add_interaction = MemoryStore.add_interaction
    def __memory_store_faiss_add_interaction(self, interaction: InteractionData):
        __memory_store_add_interaction(interaction)
        self.index.add(interaction.embedding)

    def __memory_store_faiss_retrieve(self, query_embedding, query_concepts, similarity_threshold=40, exclude_last_n=0):
        raise NotImplementedError("The FAISS-based retrieval method isn't implemented yet.")
    def __memory_store_faiss_retrieve_from_semantic_memory(self, query_embedding_norm):
        raise NotImplementedError("The FAISS-based retrieval from semantic memory method isn't implemented yet.")
    def __memory_store_faiss_cluster_interactions(self):
        raise NotImplementedError("The FAISS-based clustering method isn't implemented yet.")

    # Monkey-patch faiss in MemoryStore for speed
    MemoryStore.__init__                        = __memory_store_faiss___init__
    MemoryStore.add_interaction                 = __memory_store_faiss_add_interaction
    MemoryStore.retrieve                        = __memory_store_faiss_retrieve
    MemoryStore.retrieve_from_semantic_memory   = __memory_store_faiss_retrieve_from_semantic_memory
    MemoryStore.cluster_interactions            = __memory_store_faiss_cluster_interactions
