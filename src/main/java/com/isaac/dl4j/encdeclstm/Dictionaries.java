package com.isaac.dl4j.encdeclstm;

import java.io.IOException;
import java.util.*;

import static com.isaac.dl4j.encdeclstm.EncoderDecoderLSTM.obtainFilePath;

class Dictionaries {
    /** this number of most frequent words will be used, unknown words (that are not in the dictionary) are replaced with <unk> token */
    private static final int MAX_DICT = 20000;

    /** Dictionary that maps words into numbers. */
    static final Map<String, Double> dict = new HashMap<>();
    /** Reverse map of {@link #dict}. */
    static final Map<Double, String> revDict = new HashMap<>();
    /**
     * The contents of the corpus. This is a list of sentences (each word of the
     * sentence is denoted by a {@link Double}).
     */
    static final List<List<Double>> corpus = new ArrayList<>();

    static void createDictionary(String CORPUS_FILENAME, int ROW_SIZE) throws IOException {
        double idx = 3.0;
        dict.put("<unk>", 0.0);
        revDict.put(0.0, "<unk>");
        dict.put("<eos>", 1.0);
        revDict.put(1.0, "<eos>");
        dict.put("<go>", 2.0);
        revDict.put(2.0, "<go>");
        String CHARS = "-\\/_&" + CorpusProcessor.SPECIALS;
        for (char c : CHARS.toCharArray()) {
            if (!dict.containsKey(c)) {
                dict.put(String.valueOf(c), idx);
                revDict.put(idx, String.valueOf(c));
                ++idx;
            }
        }
        System.out.println("Building the dictionary...");
        CorpusProcessor corpusProcessor = new CorpusProcessor(obtainFilePath(CORPUS_FILENAME), ROW_SIZE, true);
        corpusProcessor.start();
        Map<String, Double> freqs = corpusProcessor.getFreq();
        Set<String> dictSet = new TreeSet<>(); // the tokens order is preserved for TreeSet
        // tokens of the same frequency fall under the same key, the order is reversed so the most frequent tokens go first
        Map<Double, Set<String>> freqMap = new TreeMap<>((o1, o2) -> (int) (o2 - o1));
        for (Map.Entry<String, Double> entry : freqs.entrySet()) {
            // tokens of the same frequency would be sorted alphabetically
            Set<String> set = freqMap.computeIfAbsent(entry.getValue(), k -> new TreeSet<>());
            set.add(entry.getKey());
        }
        int cnt = 0;
        dictSet.addAll(dict.keySet());
        // get most frequent tokens and put them to dictSet
        for (Map.Entry<Double, Set<String>> entry : freqMap.entrySet()) {
            for (String val : entry.getValue()) {
                if (dictSet.add(val) && ++cnt >= MAX_DICT) break;
            }
            if (cnt >= MAX_DICT) break;
        }
        // all of the above means that the dictionary with the same MAX_DICT constraint and made from the same source file will
        // always be the same, the tokens always correspond to the same number so we don't need to save/restore the dictionary
        System.out.println("Dictionary is ready, size is " + dictSet.size());
        // index the dictionary and build the reverse dictionary for lookups
        for (String word : dictSet) {
            if (!dict.containsKey(word)) {
                dict.put(word, idx);
                revDict.put(idx, word);
                ++idx;
            }
        }
        System.out.println("Total dictionary size is " + dict.size() + ". Processing the dataset...");
        corpusProcessor = new CorpusProcessor(obtainFilePath(CORPUS_FILENAME), ROW_SIZE, false) {
            @Override
            protected void processLine(String lastLine) {
                List<String> words = new ArrayList<>();
                tokenizeLine(lastLine, words, true);
                // modify to following one, since wordsToIndexes(words) may return empty, cause "Invalid Shape Error" in CorpusIterator
                // Details are shown in the error.log located at `resources/encdec/error.log`
                //corpus.add(wordsToIndexes(words));
                List<Double> wordsIndexes = wordsToIndexes(words);
                if (!wordsIndexes.isEmpty())
                    corpus.add(wordsIndexes);
            }
        };
        corpusProcessor.setDict(dict);
        corpusProcessor.start();
        System.out.println("Done. Corpus size is " + corpus.size());
    }
}
