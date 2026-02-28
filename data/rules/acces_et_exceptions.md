# Politique d'Accès et Exceptions – SmartParkTN

## Accès refusé : raisons possibles

1. **Véhicule blacklisté** : La plaque figure sur la liste noire (fraude, impayés, problème judiciaire).
2. **Hors horaires** : Le véhicule tente d'entrer ou sortir en dehors des heures autorisées.
3. **Abonnement expiré** : L'abonnement du véhicule n'est plus valide.
4. **Zone non autorisée** : La zone demandée n'est pas incluse dans l'abonnement.
5. **Plaque non lisible** : Qualité d'image insuffisante pour identifier le véhicule.

## Exceptions et procédures spéciales

### Ambulances et services d'urgence
- Accès immédiat et prioritaire sans contrôle horaire.
- Les caméras signalent automatiquement un événement URGENCE.
- Aucun tarif appliqué.

### Véhicules diplomatiques
- Accès autorisé sur présentation manuelle au superviseur.
- Enregistrement manuel dans le système.

### Événements spéciaux
- Le superviseur peut activer un "mode événement" permettant un accès étendu temporaire.
- Tarification spéciale configurable par l'administrateur.

### Panne de système
- En cas de panne du lecteur optique, utiliser le mode manuel (saisie clavier).
- Garder un registre papier jusqu'à rétablissement du système.

## Comment interpréter une décision du système ?

Le système SmartParkTN affiche toujours :
- La plaque détectée (avec niveau de confiance OCR)
- La catégorie identifiée (visitor, subscriber, vip, blacklist, employee, emergency)
- La décision (ALLOWED / DENIED)
- La raison de la décision (règle appliquée)

Exemple : "Accès refusé – Hors horaires autorisés (06:00–23:00). Règle: visitor_hours"
→ Le véhicule est un visiteur et a tenté d'entrer après 23h.

## Contact superviseur

- Poste superviseur : extension 201
- Urgences : extension 999
- Administration parking : admin@smartparktn.com
