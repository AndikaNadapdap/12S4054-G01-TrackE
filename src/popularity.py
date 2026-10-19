"""Popularity baseline: rank items by number of positive (rating >= 4) training interactions."""
from .common import RELEVANCE_THRESHOLD


class PopularityRecommender:
    def fit(self, train, candidates):
        pos = train[train.rating >= RELEVANCE_THRESHOLD]
        counts = pos.groupby("item_id").size().reindex(candidates, fill_value=0)
        # deterministic tie-break: higher count first, then item_id ascending
        df = counts.rename("score").rename_axis("item_id").reset_index() \
                   .sort_values(["score", "item_id"], ascending=[False, True])
        self.ranking = df.item_id.tolist()
        self.scores = dict(zip(df.item_id, df.score))
        return self

    def recommend(self, user, seen, k=10):
        out = []
        for it in self.ranking:
            if it not in seen:
                out.append(it)
                if len(out) == k:
                    break
        return out
