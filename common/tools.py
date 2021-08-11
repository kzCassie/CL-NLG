##### Train
# 1. shrink dataset while retaining the most data as possible.

##### Visualization
# How curriculum helps:
# 3. SPL
#     - self defined difficult measure (e.x. # epochs not learning from a specific example)

##### Decode

#### Eval
# - slot accuracy: need the mapping of slot <--> whether binary

# Other metrics:
# TODO: SER https://github.com/google-research/schema-guided-dialogue/tree/main/generation


"""
Technical Notes:
1. For Bucket Curriculum, the last bucket is incomplete to make sure all data are used.
   The other buckets are of exactly the same size.
"""