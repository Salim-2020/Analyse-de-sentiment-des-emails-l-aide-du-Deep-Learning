# Analyse-de-sentiment-des-emails-l-aide-du-Deep-Learning
Avant d’implémenter le réseau, les textes des emails sont transformés en vecteurs
mathématiques. Les mots du texte sont remplacés par leur index dans le dictionnaire.
Ensuite, nous ajoutons un remplissage pour rendre tous les vecteurs de même longueur.

Modele  LSTM
— Couche Embeding : Cette couche code le mot entier sous la forme d’un vecteur
flottant de 64 composants. On a pu utiliser un algorithme d’intégration spécifique
comme Word2Vec pour effectuer cette tâche, mais on a préféré laisser le réseau
déterminer quelle est la meilleure intégration pour ce dictionnaire particulier.
— Couche Dropout : Nous utilisons cette couche afin de réduire le sur-ajustement
du modèle.
— Couche LSTM : La couche récurrente, dans ce modèle nous utilisons la couche
LSTM. 
— Couche Dense : En fin, pour prédire la cible, nous ajoutons une couche entièrement
connectée (Dense) aura un neurone de sortie avec l’activation Relu


Modèle CNN
Dans le model précédent, nous remplaçons la couche LSTM par celle de la convolution1D,
l’idée derrière l’utilisation des CNN est de bénéficier de leur capacité à extraire
des fonctionnalités. Le but de l’utilisation de cette couche est d’apprendre la signification
de motifs de mots consécutifs qui peuvent avoir un sens particulier
