######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import json


class DialogActParser(object):
    """
    ## Assumption for the parser :
    ## 1. dacts are separated by the "@" token
    ## 2. slot value pairs are separated by the ";" token
    ## 3. special values specified in "resource/special_values.txt"
    ##    unify the special values by dictionary keys
    ## 4. it strips all ' or " token
    ## 5. output json format
    """
    def __init__(self, intent_sep="@", sv_sep=';', special_val_path="utils/resource/special_values.txt"):
        self.intent_sep = intent_sep
        self.sv_sep = sv_sep
        with open(special_val_path) as fin:
            self.special_values = json.load(fin)

    def parse(self, dact, keep_values=False):
        """ parse 'one' dact string.
        e.x.: "inform ( address = cheddars lane ; phone = 01223368650)"
        """
        acttype = dact.split('(')[0]
        slt2vals = dact.split('(')[1].replace(')', '').split(self.sv_sep)
        jsact = {'acttype': acttype, 's2v': []}
        for slt2val in slt2vals:
            if slt2val == '':  # no slot
                jsact['s2v'].append((None, None))
            elif '=' not in slt2val:  # no value
                slt2val = slt2val.replace('_', '').replace(' ', '')
                jsact['s2v'].append((slt2val.strip('\'\"'), '?'))
            else:  # both slot and value exist
                s, v = [x.strip('\'\"').strip() for x in slt2val.split('=')]
                s = s.replace('_', '').replace(' ', '')

                # unify the special values
                for key, vals in self.special_values.items():
                    if v in vals:
                        v = key
                if v not in self.special_values.keys() and not keep_values:  # delexicalisation
                    v = '_'
                jsact['s2v'].append((s, v))
        return jsact


class DActFormatter(object):
    """
    ## basic DAct formatter
    ## 1. abstract class for Hard and Soft subclass
    ## 2. define the basic parser command
    """

    def __init__(self, intent_sep="@", sv_sep=';', special_val_path="utils/resource/special_values.txt"):
        self.parser = DialogActParser(intent_sep, sv_sep, special_val_path)
        self.special_values = self.parser.special_values.keys()

    def format(self, dact, keep_values=False):
        raise NotImplementedError("method format() hasn't been implemented")

    def parse(self, dact, keep_values=False):
        return self.parser.parse(dact, keep_values)


class SoftDActFormatter(DActFormatter):
    """
    ## Soft DAct formatter
    ## 1. subclass of DActFormatter
    ## 2. main interface for parser/formatter
    ## 3. formatting the JSON DAct produced by DialogActParser
    ##    into a feature format fed into the network
    """
    def __init__(self, intent_sep="@", sv_sep=';', special_val_path="utils/resource/special_values.txt"):
        super().__init__(intent_sep, sv_sep, special_val_path)

    def format(self, dact, keep_values=False):
        """ output is like: [('a', 'inform'), ('address', '_1'), ('name', '_1')] """
        jsact = self.parse(dact, keep_values)
        mem = {}  # count the number of values of the same slot
        feature = []
        for s, v in jsact['s2v']:
            if s is None:  # no slot no value
                continue  # skip it
            elif v == '?':  # question case
                feature.append((s, v))
            elif v == '_':  # categories
                if s in mem.keys():  # multiple feature values
                    feature.append((s, v + str(mem[s])))
                    mem[s] += 1
                else:  # first occurance
                    feature.append((s, v + '1'))
                    mem[s] = 2
            elif v in self.special_values:  # special values
                feature.append((s, v))
        feature = [('a', jsact['acttype'])] + sorted(feature)
        return feature


if __name__ == '__main__':
    # dap = DialogActParser()
    # print(dap.parse("inform ( name = hakka restaurant ; pricerange = dont_care )", keep_values=True))
    # print(dap.parse("goodbye (  = ? )", keep_values=True))
    # print(dap.parse(" restaurant_request ( food-? )", keep_values=True))

    dadp = SoftDActFormatter()
    # dadp = HardDActFormatter()

    print( dadp.format("inform(type='restaurant';count='182';area=dont_care)"))
    print( dadp.format("reqmore()"))
    print( dadp.format("request(area)"))
    print( dadp.format("inform(name='fifth floor';address='hotel palomar 12 fourth street or rosie street')"))
    print( dadp.format("inform(name='fifth floor';address='hotel palomar 12 fourth street and rosie street')"))
    print( dadp.format("?select(food=dont_care;food='sea food')"))
    print( dadp.format("?select(food='yes';food='no')"))
    print( dadp.format("?select(battery rating=exceptional;battery rating=standard)"))
    print( dadp.format("suggest(weight range=heavy;weight range=light weight;weightrange=dontcare)"))
    print( dadp.format("?compare(name=satellite morpheus 36;warranty=1 year european;dimension=33.7 inch;name=tecra proteus 23;warranty=1 year international;dimension=27.4 inch)"))
