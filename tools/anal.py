from trie import Trie, Node
from Sentence import Sentence

def get_kmers(p_file, n_file, kmer_list):
    p_reads = list(open(p_file, "r", buffering=10000).readlines())
    p_reads = [s.strip() for s in p_reads]

    n_reads = list(open(n_file, 'r', buffering=10000).readslines())
    n_reads = [s.strip() for s in n_reads]

    p_trie, n_trie = Trie(), Trie()
    
    for read in p_reads:
        sent = Sentence(read).build_sentence(read)
        for word in sent.split():
            p_trie.add(word)

    for read in n_reads:
        sent = Sentence(read).build_sentence(read)
        for word in sent.split():
            n_trie.add(word)

if __name__ == '__main__':
    get_kmers('small_pos.txt', 'small_neg.txt', [3,10])
