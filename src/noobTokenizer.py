# Own tokenizer that tokenizes after 
class tokenizerOwn:
    def __init__(self, initVocab:list[str]):
        self.curID = 2 
        self.vocab = {"[WS]": 0, "[UKN]": 1}
        [self.addVocab(inTok) for inTok in initVocab]
        self.rev_vocab = {v: s for s,v in self.vocab.items()}
        
    
    def addVocab(self, input: str):
        self.vocab.update({input: self.curID})
        self.rev_vocab = {v: s for s, v in self.vocab.items()}
        self.curID += 1
    
    def addTextVocab(self, input: str):
        tokens = self.tokenize(input)
        for i in tokens:
            self.addVocab(i)

    def tokenize(self, input: str):
        # Diese funktion kann nach beliebig editiert werden, stand jetzt nur nach whitespaces
        return input.strip().split()


    def encoding(self, input: str):
        tokens = self.tokenize(input)
        return [self.vocab.get(tok, "[UKN]") for tok in tokens]
    
    def decoder(self, input:list[int]):
        return [self.rev_vocab.get(tokid, self.vocab.get("[UKN]")) for tokid in input]
    
tokenizer = tokenizerOwn(["test", "tree"])
test = "This is a test"
tokenizer.addTextVocab(test)
print(tokenizer.vocab)
print(tokenizer.rev_vocab)
encodedTokens = tokenizer.encoding(test+ " ts"+ " tree")
print(tokenizer.decoder(encodedTokens))
print(encodedTokens)
