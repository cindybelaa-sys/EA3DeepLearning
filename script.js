// script.js

// Globale Variablen
let model;
let tokenizer;
let isAutoGenerating = false;
let autoGenerateInterval;

// Initialisierung beim Laden
window.addEventListener("load", async () => {
    // Lade Modell und Tokenizer
    await loadModelAndTokenizer();
});

// Modell und Tokenizer laden
async function loadModelAndTokenizer() {
    // Beispiel-Pfad anpassen:
    model = await tf.loadLayersModel("model/model.json");
    console.log("Modell geladen.");

    const vocabData = await fetch("model/vocab.json").then(res => res.json());
    tokenizer = vocabData;
    console.log("Tokenizer geladen.");
}

// Eingabetext verarbeiten und vorhersagen
async function predictNextWords(prompt, topK = 5) {
    if (!model || !tokenizer) {
        alert("Modell oder Tokenizer noch nicht geladen.");
        return;
    }

    const tokens = prompt.trim().split(/\s+/);
    const inputIndices = tokens.map(word => tokenizer.word_index[word] || 0);

    let inputTensor = tf.tensor([inputIndices], [1, inputIndices.length]);

    // Modellvorhersage
    let prediction = model.predict(inputTensor);
    prediction = prediction.squeeze();

    // Sampling (Hilfsfunktion aus utils.js verwenden!)
    const sampledIndex = sampleFromTopK(prediction, topK);

    const nextWord = tokenizer.index_word[sampledIndex] || "<UNK>";
    const topKWords = Array.from(prediction.dataSync())
        .map((prob, i) => ({ word: tokenizer.index_word[i] || "<UNK>", prob }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, topK);

    return { nextWord, topKWords };
}

// UI aktualisieren
function updatePredictionUI(topKWords) {
    const listDiv = document.getElementById("prediction-list");
    listDiv.innerHTML = "";

    topKWords.forEach(item => {
        const button = document.createElement("button");
        button.textContent = `${item.word} (${(item.prob * 100).toFixed(2)}%)`;
        button.addEventListener("click", () => {
            addWordToPrompt(item.word);
            triggerPrediction();
        });
        listDiv.appendChild(button);
    });
}

// Wort zum Prompt hinzufügen
function addWordToPrompt(word) {
    const textarea = document.getElementById("input-text");
    textarea.value = textarea.value.trim() + " " + word;
}

// "Vorhersage" Button
document.getElementById("predict-button").addEventListener("click", triggerPrediction);

async function triggerPrediction() {
    const textarea = document.getElementById("input-text");
    const prompt = textarea.value.trim();

    if (prompt.length === 0) return;

    const { nextWord, topKWords } = await predictNextWords(prompt);
    updatePredictionUI(topKWords);
}

// "Weiter" Button: Wahrscheinlichstes Wort anhängen
document.getElementById("continue-button").addEventListener("click", async () => {
    const textarea = document.getElementById("input-text");
    const prompt = textarea.value.trim();

    const { nextWord } = await predictNextWords(prompt, 1);
    addWordToPrompt(nextWord);
});

// "Auto" Button: Automatisches Generieren starten
document.getElementById("auto-button").addEventListener("click", () => {
    if (isAutoGenerating) return;

    isAutoGenerating = true;
    let count = 0;

    autoGenerateInterval = setInterval(async () => {
        if (!isAutoGenerating || count >= 10) {
            clearInterval(autoGenerateInterval);
            isAutoGenerating = false;
            return;
        }
        const textarea = document.getElementById("input-text");
        const prompt = textarea.value.trim();

        const { nextWord } = await predictNextWords(prompt, 1);
        addWordToPrompt(nextWord);
        count++;
    }, 1000);
});

// "Stopp" Button
document.getElementById("stop-button").addEventListener("click", () => {
    clearInterval(autoGenerateInterval);
    isAutoGenerating = false;
});

// "Reset" Button
document.getElementById("reset-button").addEventListener("click", () => {
    document.getElementById("input-text").value = "";
    document.getElementById("prediction-list").innerHTML = "";
    clearInterval(autoGenerateInterval);
    isAutoGenerating = false;
});
