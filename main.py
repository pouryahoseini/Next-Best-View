from src import next_best_view as nbv


# Create an instance of next best view class
next_best_view = nbv.NextBestView()

# Train the vision classifiers
next_best_view.train_vision()

# Generate scores and probabilities of each frontal, side, and tile images
next_best_view.generate_scores()

# Run next best view and evaluate it
next_best_view.evaluate_next_best_view()
