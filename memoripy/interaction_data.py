#interaction_data.py
# Apache 2.0 license, Created by Awilen Bernkastel

from sklearn.preprocessing import normalize

class InteractionData:
    def __init__(self):
        self.id = None
        self.prompt = None
        self.output = None
        self.embedding = None
        self.timestamp = None
        self.last_accessed = None
        self.access_count = None
        self.decay_factor = None
        self.concepts = None
        self.normalized_embedding = None

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
