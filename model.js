function createModel(vocabSize) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 64 }));
  model.add(tf.layers.lstm({ units: 100, returnSequences: true }));
  model.add(tf.layers.lstm({ units: 100 }));
  model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function trainModel(model, sequences, vocabSize) {
  const inputs = [];
  const labels = [];

  for (const [w1, w2, label] of sequences) {
    inputs.push([w1, w2]);
    const oneHotLabel = Array(vocabSize).fill(0);
    oneHotLabel[label] = 1;
    labels.push(oneHotLabel);
  }

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(labels);

  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'acc'],
      { callbacks: ['onEpochEnd'] }
    ),
  });
}
