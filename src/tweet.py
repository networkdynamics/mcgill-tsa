"""
Simple class for storing tweet objects.
"""
import numpy as np

# globals
MAJORITY_RULE = 'majority'
MORE_COMPLICATED = 'complicated'
SOFTMAX = 'softmax'
COMPLICATED = 'com'

# tweet class
class Tweet(object):
    def __init__(self, tid, text, topic):
        self.tid = tid # tweet id, note that this is in bad format, but real TIDs are in unannotated csv
        self.orig_text = text # actual tweet text
        self.topic = topic
        self.labelling = None # this is a dictionary of labels
        self.corrected_tokens = []
        self.uncorrected_tokens = []
        self.features = {} # stores feature subsets

    def __repr__(self):
        return ("<Tweet>\n"
                "  TOPIC: {}\n"
                "  Labelling: {}\n"
                "  Original text:\n    {}\n"
                "  Preprocessed tokens:\n    {}\n"
                "</Tweet>").format(
                    self.topic,
                    self.labelling,
                    self.orig_text,
                    self.corrected_tokens,
                )

    def get_agreement(self):
        total = sum(self.labelling.values())
        max_count = max(self.labelling.values())
        return max_count / float(total)

    def get_labelling(self, option):
        label_counts = [self.labelling[label] for label in LABELS]
        total = sum(self.labelling.values())
        max_count = max(label_counts)
        agreement = max_count / float(total)

        # softmax normalizes the labelling
        if option == SOFTMAX:
            return list(map(lambda x: x / total, label_counts))

        # if the labelling has less than 51% agreement, return complicated.
        elif option == MAJORITY_RULE:
            if agreement > 0.50:
                return max(self.labelling, key=self.labelling.get)
            else:
                return COMPLICATED

        # if the labelling has less than 80% agreement, return complicated.
        elif option == MORE_COMPLICATED:
            if agreement >= 0.80:
                return max(self.labelling, key=self.labelling.get)
            else:
                return COMPLICATED

    def get_feature_vector(self, selected_feats):
        all_feats = np.array([])
        for feat, vector in self.features.items():
            if feat in selected_feats:
                all_feats = np.concatenate((all_feats, vector))
        return all_feats
