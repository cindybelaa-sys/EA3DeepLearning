// utils.js

/**
 * Samplet ein Wort basierend auf Wahrscheinlichkeitsverteilung (top-k Sampling)
 * @param {Tensor} probs - Tensor mit Wahrscheinlichkeiten (1D)
 * @param {number} k - top-k Auswahl
 * @returns {number} Index des gewählten Wortes
 */
function sampleFromTopK(probs, k = 5) {
    const probsData = probs.dataSync();
    const sorted = Array.from(probsData)
        .map((prob, i) => ({ index: i, prob }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, k);

    const total = sorted.reduce((sum, item) => sum + item.prob, 0);
    const norm = sorted.map(item => ({ ...item, prob: item.prob / total }));

    let r = Math.random();
    let acc = 0;
    for (const item of norm) {
        acc += item.prob;
        if (r < acc) return item.index;
    }
    return norm[0].index; // fallback
}

/**
 * Berechnet Top-k Accuracy
 * @param {Tensor} yTrue - Tensor mit echten Wort-Indices
 * @param {Tensor} yPred - Tensor mit Wahrscheinlichkeiten
 * @param {number} k - Top-k
 * @returns {number} Accuracy
 */
function topKAccuracy(yTrue, yPred, k) {
    const trueIndices = yTrue.dataSync();
    const predData = yPred.arraySync();
    let correct = 0;

    for (let i = 0; i < trueIndices.length; i++) {
        const sorted = predData[i]
            .map((p, j) => ({ idx: j, p }))
            .sort((a, b) => b.p - a.p)
            .slice(0, k)
            .map(item => item.idx);

        if (sorted.includes(trueIndices[i])) {
            correct++;
        }
    }
    return correct / trueIndices.length;
}
