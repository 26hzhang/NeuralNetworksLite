package com.isaac.examples;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2VecRawTextExample {

	private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

	public static void main(String[] args) throws Exception {
		// Gets Path to Text file
		String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
		log.info("Load & Vectorize Sentences....");
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(filePath);
		// Split on white spaces in the line to get words
		TokenizerFactory t = new DefaultTokenizerFactory();
        /*
         * CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
         * So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
         * Additionally it forces lower case for all tokens.
         */
		t.setTokenPreProcessor(new CommonPreprocessor());
		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder()
				.minWordFrequency(5)
				.iterations(1)
				.layerSize(100)
				.seed(42)
				.windowSize(5)
				.iterate(iter)
				.tokenizerFactory(t)
				.learningRate(0.025)
				.minLearningRate(1e-3)
				//.negativeSample(10)
				.build();
		log.info("Fitting Word2Vec model....");
		vec.fit();
		// Write word vectors to file
		log.info("Writing word vectors to text file....");
		WordVectorSerializer.writeWord2VecModel(vec, "src/main/resources/W2VModel.txt");
		// Load word vectors to Word2Vec
		Word2Vec w2v = WordVectorSerializer.readWord2VecModel("src/main/resources/W2VModel.txt");
		// Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
		log.info("Closest Words:");
		//Collection<String> lst = vec.wordsNearest("day", 10);
		Collection<String> lst = w2v.wordsNearest("day", 10);
		System.out.println("10 Words closest to 'day': " + lst);

		double cosSim = w2v.similarity("day", "night");
		System.out.println(cosSim);
	}
}
