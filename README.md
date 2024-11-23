# VisionControlAero
Étude et développement d’un système de contrôle basé sur la vision pour les produits aéronautiques : 

Ce projet a pour objectif de développer un système de contrôle qualité basé sur la vision par ordinateur, en utilisant des technologies avancées d’apprentissage profond. L’objectif est de permettre la détection, la classification et l’analyse des composants des produits aéronautiques avec précision.

Le système s'appuie sur la bibliothèque Detectron2, permettant une segmentation d'objets performante, ainsi que sur des calculs de distances entre les composants pour garantir leur positionnement correct. Une interface utilisateur intuitive a été développée pour afficher les résultats.

Fonctionnalités
- Détection et segmentation des objets :
Identification précise des composants des produits.

- Calcul des distances :
Analyse du placement des composants pour vérifier leur conformité.

- Interface utilisateur :
Visualisation des résultats d'inspection via une interface web.

- Technologies utilisées
Apprentissage profond : Detectron2
Développement web : Flask, HTML, CSS, JavaScript
Langage de programmation : Python

- ## L'interface de mon projet
![L'interface de mon projet](https://github.com/Yasminekawiche/VisionControlAero/blob/main/photo2.jpeg)

L'interface de mon projet affiche la pièce segmentée avec les détails des objets détectés. Elle indique si la pièce est conforme aux exigences du cahier des charges : 'OK' si les objets sont présents avec le nombre attendu.
