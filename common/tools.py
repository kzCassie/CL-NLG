# #TODO: name cached data file
# directory, tail = os.path.split(file_path)
#         filename = os.path.splitext(tail)[0]
#         cached_features_file = os.path.join("data_cached", directory,
#                                             args.model_type + '_encdec_' + str(enc_dec).lower() + '_seqlen_' + str(
#                                                 max_seq) + '_' +
#                                             filename + '.bin')

##### Train
# TODO: add validation file


##### Visualization



# How curriculum helps:
# 2. new data
# 3. SPL
#     - self defined difficult measure (e.x. # epochs not learning from a specific example)



##### Decode
# Need to combine evaluate, evaluator with decode
# /combine GentScorer.py, slot accuracy
"""
The hypothesis contains 0 counts of 3-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
"""


# Other metrics:
# TODO: SER https://github.com/google-research/schema-guided-dialogue/tree/main/generation
# https://github.com/mjpost/sacrebleu
# don't do: Translation error rate (TER)


"""
Technical Notes:
1. For Bucket Curriculum, the last bucket is incomplete to make sure all data are used.
   The other buckets are of exactly the same size.
2. dev set: same as train set
"""


"""
3. separate deocoding and evaluating
    - sacreBLEU
    - slot accuracy
"""