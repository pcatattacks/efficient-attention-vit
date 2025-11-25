# Implementation Plan

1. Load ImageNet-100 dataset
    1. write utility functions for plotting an image for sanity checking the dataset
    2. utility functions for normalizing / denormalizing image - make a log of the preprocessing steps involved for the pretrained vision transformer, so we can undo the preprocessing to visualize the image.
2. Implement the base implementation of the VIT BACKBONE 
3. Set up fine-tuning pipeline:
    1. Write fine tuning training loop (reusable function with different models able to be fed in)
    2. Write model evaluation functions for:
        1. Accuracy (top-k, k should be a parameter)
        2. GPU memory usage (see whats available in pytorch or other libraries to measure this - perhaps this could be a decorator on top of the training loop function)
        3. FLOPs (again, see what's available for this - could use the decorater pattern again.)
        4. Parameter counts should be easily available my using the PyTorch module.named_parameters() or module.parameters() function, and we can call len() on it.
    3. Set up tensorboard for monitoring model training, looking at graphs etc.
4. Find pre-implemented Linformer, Performer, Nyströmformer attention variants of vision transformers if available
    1. If not available, implement by creating interchangeable attention blocks that can be changed by passing in an argument into a module constructor.
    2. Carefully document tensor shapes in comments for clarity.
5. Fine tune each vision transformer variant while using evaluation functions to track metrics. Plot graphs on tensorboard for comparison. 
    1. Ensure that training pipeline runs successfully without fine-tuning too much, since we can train and graph all variants' performance together once the hybrid attention mechanism is implemented. This is just a sanity check to ensure we don't get runtime errors / training isn't unstable.
6. Implement hybrid attention mechanism, documenting intuition and how it's implemented. What aspects of atrous attention are taken and what aspects of linear attention / kernel tricks are used?
7. Run final fine-tuning for all models, graph on tensorboard and compare. Ensure you save models using save_weight_dict() so we don't have to train models again.
