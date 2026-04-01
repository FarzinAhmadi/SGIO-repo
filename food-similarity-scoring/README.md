# Food Similarity Scorer

Initial prototype of a food product search over the [Open Food Facts](https://world.openfoodfacts.org/) dataset.

The current pipeline combines:

- semantic similarity (FAISS embedding retrieval + reranker),
- optional macronutrient profile scoring (per 100g targets),
- optional metadata scoring (categories/tags/food groups/nutriscore/nova).

Optional fields are user-driven: if a field is provided it contributes to ranking; blank fields are ignored.
