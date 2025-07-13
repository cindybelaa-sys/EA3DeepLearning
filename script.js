function predictNextWords(model, inputWords, wordToIndex, indexToWord, vocabSize, topK = 5) {
  const inputIdx = inputWords.map(w => wordToIndex[w.toLowerCase()]);
  const xs = tf.tensor2d([inputIdx]);
  const preds = model.predict(xs);
  const probs = preds.dataSync();

  // Top K Wörter
  const topIndices = [...probs.keys()]
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, topK);

  return topIndices.map(i => ({ word: indexToWord[i], prob: probs[i] }));
}
