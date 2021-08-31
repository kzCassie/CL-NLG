class MultiRefSacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.multiple_refs = defaultdict(set)
        self.sys = []
        self.refs = []
        self.srcs = []

    def add_string_src(self, ref, pred, src):
        key = src  # for src without value
        self.multiple_refs[key].add(ref)
        self.sys.append(pred)
        self.srcs.append(key)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        scores = []
        for j in range(len(self.srcs)):
            key = self.srcs[j]
            scores.append(self.sacrebleu.corpus_bleu([self.sys[j]], [[x] for x in list(self.multiple_refs[key])],
                                                     force=True).score)
        return statistics.mean(scores)


# To use it you first create the scorer object, populate the src-refs dictionary and then
# call the result_string method to compute BLEU per each input and return the average.


scorer = MultiRefSacrebleuScorer()
for i in range(len(src)):
    scorer.add_string_src(trg[i], hyp[i], src[i])
score = scorer.result_string()

