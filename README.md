# LSTM Language Model mit TensorFlow.js

## 📌 Projektbeschreibung
Dieses Projekt ist ein einfaches autoregressives Sprachmodell, das mit einem LSTM-Netzwerk in TensorFlow.js trainiert wurde. Es erlaubt dem Nutzer, interaktiv ein deutsches Textstück zu generieren, indem jeweils das nächste wahrscheinliche Wort vorhergesagt wird.

## 🔧 Projektstruktur


## 💡 Funktionen

- Wortvorhersage mit LSTM
- Interaktive Buttons:
  - 🔍 Vorhersage
  - ➡ Weiter (wahrscheinlichstes Wort)
  - 🔁 Auto (max. 10 Wörter)
  - ⏹ Stop
  - 🔄 Reset
- Visualisierung der Top-k-Wahrscheinlichkeiten

## 🧠 Modellarchitektur

- 2 LSTM-Layer, je 100 Units
- Dense Softmax-Ausgabe
- Optimizer: Adam (LR = 0.01)
- Loss: Categorical Crossentropy
- Batch Size: 32

## 🧪 Experimente & Evaluation

- Accuracy bei Top-k: k=1,5,10,20,100
- Perplexity-Berechnung
- Training auf Subset deutscher Texte (z.B. Gutenberg, Bundestag)

## 💻 Voraussetzungen

- Lokaler Webserver (z. B. VS Code + Live Server)
- Chrome oder Firefox
- Internetzugang (für CDN von TensorFlow.js)

## 🔗 Quellen

- TensorFlow.js Demos: https://www.tensorflow.org/js/demos
- LSTM-Textgenerierung: https://github.com/seyedsaeidmasoumzadeh/Predict-next-word
