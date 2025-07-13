const rawText = `Ein Beispieltext zum Trainieren eines einfachen LSTM-Netzes mit TensorFlow.js`;

function preprocessText(text) {
  const words = text.toLowerCase().split(" ");
  const vocab = [...new Set(words)];
  const wordToIndex = Object.fromEntries(vocab.map((w, i) => [w, i]));
  const indexToWord = Object.fromEntries(vocab.map((w, i) => [i, w]));

  // Training Samples (Input: 2 Wörter, Output: nächstes Wort)
  const sequences = [];
  for (let i = 0; i < words.length - 2; i++) {
    sequences.push([
      wordToIndex[words[i]],
      wordToIndex[words[i + 1]],
      wordToIndex[words[i + 2]],
    ]);
  }

  return { sequences, wordToIndex, indexToWord, vocabSize: vocab.length };
}
