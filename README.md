

<h1> NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE </h1>

<h3> Contexte: </h3>
Ce projet propose une explication simple mais surtout en français du papier "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE" de Dzmitry BAHDANAU, Kyung Hyun CHO et de Yoshua BENGIO. Il apporte aussi une implémentation de ce papier en adaptant une version pytorch revisitée du tutoriel sur la traduction automatique (de textes espagnol vers des textes en anglais de tensorflow) à l'aide de réseau de neurones. Vous pouvez trouver la version originale du tutoriel <a href="https://www.tensorflow.org/tutorials/text/nmt_with_attention">ici</a>. Quant au papier vous pouvez le trouver <a href="https://arxiv.org/abs/1409.0473">ici</a>.

<h3> Ce que vous pouvez retenir du papier: </h3>

La traduction automatique de textes grâce à des réseaux de neurones (Natural Machine Translation en anglais ou NMT en abrégé) est une approche de plus en plus en vogue. Son but est de permettre d'obtenir une traduction de qualité avec un seul composant (grand réseau de neurones) qui lit à la fois le texte d'entrée et le traduit dans la langue d'arrivée. Ce qui se différencie de la traduction statistique traditionnelle. 

Le papier propose une architecture performante de réseau de neurones dite: encodeur-attention-décodeur qui permet lors de la traduction d'un mot de se focaliser sur les groupes de mots dans la séquence d'entrée les plus importants définissant un contexte pour le mot à traduire. Ce méchanisme est appelé méchanisme d'attention et permet d'obtenir des performances remarquables même en présence de très longues phrases à traduire.

Supposons que nous avons une séquence en espagnol (phrase source) que nous voulons traduire vers l'anglais(phrase cible). Dans les architectures traditionnelles d'encodeur-décodeur précédent ce papier, l'encodeur lit la phrase source qui est représentée mathématiquement comme la séquence
<div> <img src="https://latex.codecogs.com/svg.latex?X&space;=&space;(X_{1},&space;X_{2},...,&space;X_{T_{X}})" title="X = (X_{1}, X_{2},..., X_{T_{X}})" /> </div>
avec
<div> <a href="https://www.codecogs.com/eqnedit.php?latex=T_{x}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?T_{x}" title="T_{x}" /></a> </div>
étant la longueur de la phrase (le nombre de mots). Et la représente en un contexte C. Généralement on utilise un RNN (réseau de neurone récurrent) qui produit les sorties:
<div>
<img src="https://latex.codecogs.com/svg.latex?h_{t}&space;=&space;f(X_{t},&space;h_{t-1})" title="h_{t} = f(X_{t}, h_{t-1})" />
</div>

et le contexte:

<img src="https://latex.codecogs.com/svg.latex?C&space;=&space;q(\left&space;\{&space;h_{1},&space;...,&space;h_{T}&space;\right&space;\})" title="C = q(\left \{ h_{1}, ..., h_{T} \right \})" />

f et q sont des fonctions non-linéaires telles que:
