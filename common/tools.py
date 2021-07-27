# TODO: dataloader
# remove last batch, so valid avg metric score?

# #TODO: name cached data file
# directory, tail = os.path.split(file_path)
#         filename = os.path.splitext(tail)[0]
#         cached_features_file = os.path.join("data_cached", directory,
#                                             args.model_type + '_encdec_' + str(enc_dec).lower() + '_seqlen_' + str(
#                                                 max_seq) + '_' +
#                                             filename + '.bin')
#
##### Train
# TODO: use tensor board for visualization
# TODO: add patience
# TODO: add gpt2?
# TODO: validation set: which set to use? same as train set?
#       TODO early stopping / info max epoch reached


##### Decode
#TODO:decode - topk / topp / beam size
#TODO: evaluate.py wrong as it gives tgt utterance to decoder when evaluating loss?
#     Need to combine evaluate, evaluator with decode
# /combine evaluator.py, slot accuracy


# curriculum
# TODO: small last bucket?
#  tqdm(curriculumns)


##### Visualization
# TODO:  loss after each curriculum
# organize results


# difficulty:
# Hierarchical ranking
# number of intentions
# number of slots

# segment by # of bucket:

# How curriculum helps:
# keep record of training time
# history of losses
# number of times example used / steps

# 1. format resutls
# 2. new data
# 3. SPL
#     - self defined difficult measure (e.x. # epochs not learning from a specific example)


# metric:
# https://github.com/google-research/schema-guided-dialogue/tree/main/generation
# giuliozhou g80054706 to Everyone (3:45 PM)
# https://github.com/mjpost/sacrebleu
# Gerasimos Lampouras g00496802 to Everyone (3:47 PM)
# METERO/TER/BLUERT
# Don't do TER
# Do BLEURT and METEOR