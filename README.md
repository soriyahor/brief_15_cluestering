# SIMPLON DEV IA | Brief 15
## AJAX clustering - Exposition de plusieurs modèles et Paramétrage via une API-REST:

Groupe : Soriya, Cyril

### Contexte du projet
Le but de ce brief est de créer une application intégrant :
- un back-end proposant au moins 3 modèles de clustering sur la base de données du "Mall Customer Segmentation",
- une interface Web permettant de choisir dynamiquement entre différents modèles avec AJAX et d'afficher un indicateur de performance associé,
- une spécification de l'API qui va expose ce back-end


### Procédure 

-> Importer la base de données, créer et stocker dans un fichier "data" dans le fichier "back": 
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python


-> Lancer l'API :
```bash
python api.py
```

-> Lancer l'API contenant les deux modèles de Clustering :
```bash
python models.py
```

-> Ouvrez "templates/index.html" dans votre navigateur web pour accéder à l'interface :
Vous pouvez sélectionner le modèle voulu et le nombres de clusters désirés.

Note - au-dela de 4 clusters une erreur peut subvenir selon votre connection et votre surface d'application :

```bash
httpx.ReadTimeout
```
### Dockerisation

Déployez le front, le back et le réseau Docker pour permettre la communication entre les conteneurs avec le fichier docker-compose : 

```bash
docker-compose up
```

(à éxécuter dans le dossier racine)


## Rapports SpecOps :

Le projet est constitué de deux "micro-services" : le premier est l'API permettant l'échange de données avec le modèle et la saisie de formulaire HTML, ces données sont envoyés via une API-REST au deuxième, le modèle qui execute sa prédiction sur la data fournie en local, est évalué et génère un graphique du clustering réalisé. Le graphique et les métriques du modèle sont renvoyés à l'interface UI via une deuxième API-REST.

### L'API : api.py
L'interface propose de l'intéractivité grâce à AJAX qui permet de requêter en Javascript la page web sans la recharger.<br>
Le choix du modèle et le choix du paramètres "nombre de clusters" sont proposés dans l'UI et sont trasnmis au modèle via une requête API "POST".<br>
Les deux modèles ont chacun leurs "Endpoints" indépendants, l'API suivant le modèle sélectionné envoie une requête à l'un des deux.<br>
Le réseau d'API est hébergé par un serveur "Uvicorn".

### Le Modèle : models.py

Nous avons utilisé deux modèles de clusteting de la librarie "sklearn.cluster" : le "Kmeans" et le "Agglomerative Clustering".<br>

Le modèle crée une image du graphique de sauvegarde dans un dossier "img", le graphique est codé en binaire avec "BytesIO" et en base 64 pour être stockée en Buffer, puis envoyé, avec les autres informations du graphique :        

```bash
        'model_name': 'Kmeans',
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'plot_image': plot_base64
```

Le formatage des données est commun aux deux modèles et réalisé avec la librarie "pandas".<br>

Les deux modèles ont chacun leurs "Endpoints" indépendants contenant une fonction asynchrone réalisant le "processing" du modèle,calculant les métriques et la création du graphique avec "matplotlib.pyplot".<br>

Les métriques Silhouette et Davies-Bouldin sont des indices récurrent dans l'analyse de performance de modèle de clustering.<br>



### Repartition des tâches

Soriya a travaillé sur le front : Création de l'UI et de l'API permettant l'échange de données avec le modèle.<br>

Cyril a travaillé sur le back : Création du script du modèle et intégration des fonctions de prédiction et de graphiques, Implémentation des end-points.<br>

