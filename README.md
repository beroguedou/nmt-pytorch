

<h1> NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE </h1>

<h3> Contexte: </h3>
<p>
Ce projet propose une explication simple mais surtout en français du papier "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE" de Dzmitry BAHDANAU, Kyung Hyun CHO et de Yoshua BENGIO. Il apporte aussi une implémentation de ce papier en adaptant une version pytorch revisitée du tutoriel sur la traduction automatique (de textes espagnol vers des textes en anglais de tensorflow) à l'aide de réseau de neurones. Vous pouvez trouver la version originale du tutoriel <a href="https://www.tensorflow.org/tutorials/text/nmt_with_attention">ici</a>. Quant au papier vous pouvez le trouver <a href="https://arxiv.org/abs/1409.0473">ici</a>.
</p>

<h3> Ce que vous pouvez retenir du papier: </h3>
<p>
La traduction automatique de textes grâce à des réseaux de neurones (Natural Machine Translation en anglais ou NMT en abrégé) est une approche de plus en plus en vogue. Son but est de permettre d'obtenir une traduction de qualité avec un seul composant (grand réseau de neurones) qui lit à la fois le texte d'entrée et le traduit dans la langue d'arrivée. Ce qui se différencie de la traduction statistique traditionnelle. 

Le papier propose une architecture performante de réseau de neurones dite: encodeur-attention-décodeur qui permet lors de la traduction d'un mot de se focaliser sur les groupes de mots dans la séquence d'entrée les plus importants définissant un contexte pour le mot à traduire. Ce méchanisme est appelé méchanisme d'attention et permet d'obtenir des performances remarquables même en présence de très longues phrases à traduire.

Supposons que nous avons une séquence en espagnol (phrase source) que nous voulons traduire vers l'anglais(phrase cible). Dans les architectures traditionnelles d'encodeur-décodeur précédent ce papier, l'encodeur lit la phrase source qui est représentée mathématiquement comme la séquence
<img src="https://latex.codecogs.com/svg.latex?X&space;=&space;(X_{1},&space;X_{2},...,&space;X_{T_{X}})" title="X = (X_{1}, X_{2},..., X_{T_{X}})" />
avec
<img src="https://latex.codecogs.com/svg.latex?T_{X}" title="T_{X}" />
étant la longueur de la phrase (le nombre de mots). Et la représente en un contexte C. Généralement on utilise un RNN (réseau de neurone récurrent) qui produit les sorties:
<img src="https://latex.codecogs.com/svg.latex?h_{t}&space;=&space;f(X_{t},&space;h_{t-1})" title="h_{t} = f(X_{t}, h_{t-1})" /> et le contexte:
<img src="https://latex.codecogs.com/svg.latex?C&space;=&space;q(\left&space;\{&space;h_{1},&space;...,&space;h_{T}&space;\right&space;\})" title="C = q(\left \{ h_{1}, ..., h_{T_{X} \right \})" /> 
f et q sont des fonctions non-linéaires telles que:  
<img src="https://latex.codecogs.com/svg.latex?f&space;=&space;LSTM" title="f = LSTM" /> et <img src="https://latex.codecogs.com/svg.latex?q(\left&space;\{&space;h_{1},&space;...&space;,h_{T}&space;\right&space;\})&space;=&space;h_{T}" title="q(\left \{ h_{1}, ... ,h_{T} \right \}) = h_{T}" />.

<img src="Image1.png"/> 

Le décodeur quant à lui est entrainé pour prédire le prochain mot <img src="https://latex.codecogs.com/svg.latex?y_{t}" title="y_{t}" /> connaissant le vecteur contexte C et et tous les mots déja prédits <img src="https://latex.codecogs.com/svg.latex?\left&space;\{&space;y_{1},&space;...&space;,y_{t-1}&space;\right&space;\}" title="\left \{ y_{1}, ... ,y_{t-1} \right \}" />. En d'autres termes la phrase cible est modélisée par une probabilité jointe sur tous ses mots qui peut s'écrire: 

<img src="https://latex.codecogs.com/svg.latex?p(y)&space;=&space;\prod_{t=1}^{T_{y}}&space;p(y_{t}/\left&space;\{&space;y_{1},&space;...&space;,y_{t-1}&space;\right&space;\},&space;C)" title="p(y) = \prod_{t=1}^{T_{y}} p(y_{t}/\left \{ y_{1}, ... ,y_{t-1} \right \}, C)" />
où  <img src="https://latex.codecogs.com/svg.latex?y&space;=&space;(y_{1},&space;...&space;,y_{T_{y}})" title="y = (y_{1}, ... ,y_{T_{y}})" />.

Avec un RNN dans le décodeur on modélise chaque probabilté conditionnelle comme : <img src="https://latex.codecogs.com/svg.latex?p(y_{t}/\left&space;\{&space;y_{1},&space;...&space;,y_{t-1}&space;\right&space;\},&space;C)&space;=&space;g(y_{i-1},&space;S_{i},&space;C)" title="p(y_{t}/\left \{ y_{1}, ... ,y_{t-1} \right \}, C) = g(y_{i-1}, S_{i}, C)" /> (le mot qui vient d'être prédit, les états cachés correspondant au mot à prédire et tout le vecteur contexte C). La fonction g est non linéaire délivrant la probabilié <img src="https://latex.codecogs.com/svg.latex?y_{t}" title="y_{t}" /> et <img src="https://latex.codecogs.com/svg.latex?S_{t}" title="S_{t}" /> étant l'état caché du RNN.

Le papier est innovant en ce qu'il propose de définir chaque probabilité conditionnelle comme  : <img src="https://latex.codecogs.com/svg.latex?p(y_{t}/\left&space;\{&space;y_{1},&space;...&space;,y_{t-1}&space;\right&space;\},&space;C_{i})&space;=&space;g(y_{i-1},&space;S_{i},&space;C)" title="p(y_{t}/\left \{ y_{1}, ... ,y_{t-1} \right \}, C_{i}) = g(y_{i-1}, S_{i}, C_{i})" /> où <img src="https://latex.codecogs.com/svg.latex?C_{i}" title="C_{i}" /> est désormais un vecteur contexte dynamique (donc non statique comme précédemment) et dépendant du mot <img src="https://latex.codecogs.com/svg.latex?y_{i}" title="y_{i}" /> à prédire.
</p>

<h4> L'encodeur: </h4>
<p>
L'encodeur est simplement un RNN bdirectionnel même si nous utiliserons un RNN à une seule direction dans ce projet. Cet encodeur "mappe" une séquence d'entrée <img src="https://latex.codecogs.com/svg.latex?X&space;=&space;(X_{1},&space;X_{2},...,&space;X_{T_{X}})" title="X = (X_{1}, X_{2},..., X_{T_{X}})" /> en une séquence d'annotations <img src="https://latex.codecogs.com/svg.latex?(h_{1},&space;h_{2},...,&space;h_{T_{X}})" title=" (h_{1}, h_{2},..., h_{T_{X}})" /> bidirectionnelle. Et chaque <img src="https://latex.codecogs.com/svg.latex?h_{i}" title="h_{i}" /> est obtenu par concaténation du forward pass et du bacward pass du RNN et contient de l'information sur toute la séquence avec un grand focus sur le i-ème élément de la séquence d'entrée.
  
</p>

<h4> Le décodeur avec méchanisme d'attention: </h4>
<p>
Les équations suivantes sont computées : <img src="https://latex.codecogs.com/svg.latex?e_{ij}&space;=&space;a\left&space;(&space;S_{i-1},&space;h_{j}&space;\right&space;)" title="e_{ij} = a\left ( S_{i-1}, h_{j} \right )" /> où "a" est une fonction d'attention qui score comment les inputs autour du j-ème élément de la séquence d'entrée et l'output de la position i matchent. <img src="https://latex.codecogs.com/svg.latex?S_{i-1}" title="S_{i-1}" /> est l'état caché du RNN avant d'émettre  <img src="https://latex.codecogs.com/svg.latex?y_{i}" title="y_{i}" /> .
<p>

<img src="https://latex.codecogs.com/svg.latex?a\left&space;(&space;S_{i-1},&space;h_{j}&space;\right&space;)&space;=&space;v_{a}^{T}tanh(W_{a}S_{i-1}&space;&plus;&space;U_{a}h_{j}])" title="a\left ( S_{i-1}, h_{j} \right ) = v_{a}^{T}tanh(W_{a}S_{i-1} + U_{a}h_{j}])" />
Les auteurs ont choisi un MLP pour des raisons de computation car le modèle est évalué <img src="https://latex.codecogs.com/svg.latex?T_{x}*T_{y}" title="T_{x}*T_{y}" /> pour chaque paire phrase source - phrase cible.
<div>
<img src="https://latex.codecogs.com/svg.latex?\alpha&space;_{ij}&space;=&space;\frac{exp(e_{ij})}{\sum_{k=1}^{T_{x}}exp(e_{ik})}" title="\alpha _{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_{x}}exp(e_{ik})}" />
</div>



<img src="https://latex.codecogs.com/svg.latex?C_{i}&space;=&space;\sum_{j=1}^{T_{x}}&space;\alpha&space;_{ij}h_{j}" title="C_{i} = \sum_{j=1}^{T_{x}} \alpha _{ij}h_{j}" /> 
<div>
<img src="https://latex.codecogs.com/svg.latex?h_{j}&space;=&space;\left&space;\lfloor&space;\overrightarrow{h_{j}};&space;\overleftarrow{h_{j}}&space;\right&space;\rfloor" title="h_{j} = \left \lfloor \overrightarrow{h_{j}}; \overleftarrow{h_{j}} \right \rfloor" />
</div>

Pour le décodeur on passe en pratique la concaténation de <img src="https://latex.codecogs.com/svg.latex?y_{i-1}" title="y_{i-1}" /> et de <img src="https://latex.codecogs.com/svg.latex?C_{i}" title="C_{i}" /> a un RNN (GRU) dont l'état est <img src="https://latex.codecogs.com/svg.latex?S_{i-1}" title="S_{i-1}" /> pour prédire <img src="https://latex.codecogs.com/svg.latex?y_{i}" title="y_{i}" />.

Selon le papier ils ont ensuite passé la sortie du RNN à une couche de maxout units puis normaliser avec un softmax pour avoir les probabilités sur l'espace défini par le vocabulaire de la langue d'arrivée.

Le maxout modèle est simplement une architecture feed-forward (comme un MLP) qui utilise une fonction d'activation  appelée maxout unit. Etant donnée une entrée <img src="https://latex.codecogs.com/svg.latex?x&space;\epsilon&space;\mathbb{R}^{d}" title="x \epsilon \mathbb{R}^{d}" /> on compute <img src="https://latex.codecogs.com/svg.latex?Z_{ij}&space;=&space;X^{T}W_{ij}&space;&plus;&space;b_{ij}" title="Z_{ij} = X^{T}W_{ij} + b_{ij}" /> (ce qui revient à passer X dans une couche dense !) puis on applique la fonction d'activation appelée maxout unit:
<div>
<img src="https://latex.codecogs.com/svg.latex?h_{i}(X)&space;=&space;max_{j\epsilon&space;[1,&space;k]}Z_{ij}" title="h_{i}(X) = max_{j\epsilon [1, k]}Z_{ij}" />
</div>

Ian GOODFELLOW et Yoshua BENGIO ont démontré que c'est un approximateur universel. Mais l'incovénient est qu'il double le nombre de paramètres donc j'utiliserai plutôt une Relu à la place.









  






