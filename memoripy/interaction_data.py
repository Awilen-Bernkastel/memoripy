#interaction_data.py
# Apache 2.0 license, Created by Awilen Bernkastel

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InteractionData:
    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', None)
        self.prompt = kwargs.get('prompt', None)
        self.output = kwargs.get('output', None)
        self.embedding = kwargs.get('embedding', None)
        self.timestamp = kwargs.get('timestamp', None)
        self.last_accessed = kwargs.get('last_accessed', None)
        self.access_count = kwargs.get('access_count', None)
        self.decay_factor = kwargs.get('decay_factor', None)
        self.concepts = kwargs.get('concepts', None)
        self.normalized_embedding = kwargs.get('normalized_embedding', None)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default_value=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default_value

    def normalize_embedding(self):
        if self.normalized_embedding is None:
            self.normalized_embedding = normalize(self.embedding)
        return self.normalized_embedding

    def update_decay_factor(self, factor):
        self.decay_factor *= factor
        return self.decay_factor

    def embedding_similarity(self, embedding):
        return cosine_similarity(embedding, self.normalize_embedding())[0][0]

    def adjusted_similarity(self, query_interaction, current_time):
        decay_rate = 0.0001 # Adjust decay rate as needed
        # Compute the cosine similarity
        # Multiply by the reinforcement factor
        # Multiply by the decay factor that's updated in the process
        return (self.embedding_similarity(query_interaction.normalize_embedding())) * \
                np.log1p(self.access_count) * \
                self.update_decay_factor(np.exp(-decay_rate * (current_time - self.last_accessed)))
