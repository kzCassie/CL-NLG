


# TODO: dataloader
# remove last batch, so valid avg metric score


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


##### Decode
#TODO:decode - topk / topp / beam size
#TODO: evaluate.py wrong as it gives tgt utterance to decoder when evaluating loss?
#     Need to combine evaluate, evaluator with decode
# implement more metrics: BLEU
# /combine evaluator.py


