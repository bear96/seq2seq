import re


SOS_token = 0
EOS_token = 1
PAD_token = 2

#tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
class PolyLang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2: "PAD"}
        self.n_words = 3  

    def addSentence(self, sentence):
        for word in sentence_to_words(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1,lang2, file_path):
    print("Reading lines...")

    # Read the file and split into lines
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])

    Factor_vocab = PolyLang(lang1)
    Expansions_vocab =  PolyLang(lang2)

    return Factor_vocab, Expansions_vocab, factors,expansions

def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)
    
##Split data into train test here
def sentence_to_words(sentence):
    return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", sentence.strip().lower())
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence_to_words(sentence)]


def tensorFromSentence(lang, sentence,device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(lang1,lang2, factors, expansions):
    input_tensor = tensorFromSentence(lang1, factors)
    target_tensor = tensorFromSentence(lang2, expansions)
    return (input_tensor, target_tensor)


from tqdm import tqdm
MAX_LENGTH = 30

def preparevocab(lang1, lang2,file_path):
    f_vocab,expans_vocab, factors,expansions = readLangs(lang1,lang2, file_path)
    print("Read %s sentence pairs" % len(factors))
    print("Building Polynomial vocabulary...")
    for i in tqdm(range(len(factors))):
        f_vocab.addSentence(factors[i])
        expans_vocab.addSentence(expansions[i])
    print("Counted words:")
    print(f_vocab.name, f_vocab.n_words)
    print(expans_vocab.name, expans_vocab.n_words)
    return f_vocab, expans_vocab,factors,expansions
