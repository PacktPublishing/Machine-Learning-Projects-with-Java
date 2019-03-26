package com.tomekl007.chapter5;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;


public class FindWordsSimilarityModel {

  private static final Logger logger = LoggerFactory.getLogger(FindWordsSimilarityModel.class);

  public static void main(String[] args) throws Exception {

    ClassPathResource resource = new ClassPathResource("raw_textual_data");
    File file = resource.getFile();

    AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

    BasicLineIterator underlyingIterator = new BasicLineIterator(file);


    SentenceTransformer transformer = tokenizeInput(underlyingIterator);

    AbstractSequenceIterator<VocabWord> sequenceIterator =
        new AbstractSequenceIterator.Builder<>(transformer).build();

    buildVocabulary(vocabCache, sequenceIterator);
    WeightLookupTable<VocabWord> lookupTable = buildWeightLookupTable(vocabCache);


    lookupTable.resetWeights(true);


    SequenceVectors<VocabWord> vectors = buildModel(vocabCache, sequenceIterator, lookupTable);

    vectors.fit();
    findSimilarities(vectors);

  }

  private static void findSimilarities(SequenceVectors<VocabWord> vectors) {
    double sim = vectors.similarity("day", "night");
    logger.info("Day/night similarity: " + sim);

    Collection<String> words = vectors.wordsNearest("day", 10);
    System.out.println("Nearest words to 'day': " + words);
  }

  private static SequenceVectors<VocabWord> buildModel(AbstractCache<VocabWord> vocabCache, AbstractSequenceIterator<VocabWord> sequenceIterator, WeightLookupTable<VocabWord> lookupTable) {
    return new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
        .minWordFrequency(5)
        .lookupTable(lookupTable)
        .iterate(sequenceIterator)
        .vocabCache(vocabCache)
        .batchSize(250)
        .iterations(1)
        .epochs(1)
        .resetModel(false)
        .trainElementsRepresentation(true)
        .trainSequencesRepresentation(false)
        .elementsLearningAlgorithm(new SkipGram<>())
        .build();
  }

  private static WeightLookupTable<VocabWord> buildWeightLookupTable(AbstractCache<VocabWord> vocabCache) {
    return new InMemoryLookupTable.Builder<VocabWord>()
        .vectorLength(150)
        .useAdaGrad(false)
        .cache(vocabCache)
        .build();
  }

  private static void buildVocabulary(AbstractCache<VocabWord> vocabCache, AbstractSequenceIterator<VocabWord> sequenceIterator) {
    VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
        .addSource(sequenceIterator, 5)
        .setTargetVocabCache(vocabCache)
        .build();

    constructor.buildJointVocabulary(false, true);
  }

  private static SentenceTransformer tokenizeInput(BasicLineIterator underlyingIterator) {
    TokenizerFactory t = new DefaultTokenizerFactory();
    t.setTokenPreProcessor(new CommonPreprocessor());

    return new SentenceTransformer.Builder()
        .iterator(underlyingIterator)
        .tokenizerFactory(t)
        .build();
  }
}
