# Fiche de révision — Single-turn, multi-turn, single-step et multi-step

## La règle la plus importante

Les mots **turn** et **step** sont employés différemment selon les articles et les frameworks.

Pour éviter toute confusion, compte toujours séparément :

1. les **messages utilisateur** ;
2. les **invocations du LLM** (générations) ;
3. les **appels d’outils** (actions) ;
4. les **interactions modèle–environnement** ;
5. l’**épisode complet**.

La question opérationnelle la plus importante est :

> **Le LLM est-il appelé une nouvelle fois après avoir reçu le résultat d’un outil ?**

- **Non** → une seule génération du modèle.
- **Oui** → trajectoire agentique avec plusieurs générations/étapes.

---

## Les unités à distinguer

| Unité | Définition |
|---|---|
| **User turn** | Un message envoyé par l’utilisateur |
| **Model turn / generation** | Une invocation du LLM |
| **Tool call / action** | Un appel d’outil demandé par le LLM |
| **Tool observation** | Le résultat renvoyé par l’outil au modèle |
| **Interaction step** | Généralement : une génération du modèle suivie de l’exécution de ses actions |
| **Episode / trajectory** | Toute la résolution d’une tâche, du prompt initial à la fin |

## Cas 1 — Réponse classique

```text
Utilisateur
    ↓
Génération 1 : réponse finale
```

- 1 message utilisateur ;
- 1 génération du LLM ;
- 0 appel d’outil ;
- épisode single-step ;
- conversation single-turn.

## Cas 2 — Plusieurs appels d’outils générés ensemble

```text
Utilisateur
    ↓
Génération 1 : [appel A, appel B, appel C]
    ↓
Exécution des trois appels
```

- 1 message utilisateur ;
- 1 génération du LLM ;
- 3 appels d’outils ;
- 1 round modèle–environnement ;
- **single-model-turn, multi-call**.

Les trois appels peuvent être générés dans la même réponse, par exemple :

```xml
<tool_call>{"name": "tool_a", "arguments": {...}}</tool_call>
<tool_call>{"name": "tool_b", "arguments": {...}}</tool_call>
```

Ce cas ne devient pas multi-turn uniquement parce qu’il contient plusieurs appels.

Certains articles peuvent néanmoins compter chaque action comme un « step » et appeler ce cas *multi-step*. C’est pour cela que **single-turn, multi-call** est une description plus précise.

## Cas 3 — Le modèle reçoit les résultats et génère de nouveau

```text
Utilisateur
    ↓
Génération 1 : rechercher l’email de Carlos
    ↓
Outil : carlos@example.com
    ↓
Génération 2 : envoyer un email à carlos@example.com
    ↓
Outil : email envoyé
    ↓
Génération 3 : réponse finale
```

- 1 message utilisateur ;
- plusieurs générations du LLM ;
- plusieurs interactions modèle–environnement ;
- trajectoire **multi-step** ;
- souvent appelée **multi-turn tool use** ou **multi-step agentic rollout**.

Elle reste cependant *single-turn* si l’auteur réserve le mot *turn* aux messages humains, car l’utilisateur n’a envoyé qu’un seul message.

## Cas 4 — Conversation humaine multi-turn

```text
Utilisateur → Assistant → Utilisateur → Assistant
```

- plusieurs messages utilisateur ;
- plusieurs tours de conversation ;
- **multi-turn conversation** au sens humain.

Ce n’est pas la même chose qu’un agent qui effectue plusieurs appels d’outils après une seule demande.

---

## Pourquoi la terminologie paraît contradictoire

Deux définitions de **turn** coexistent :

| Sens de « turn » | « Multi-turn » signifie… |
|---|---|
| Conversation humaine | plusieurs messages utilisateur |
| Interaction agentique | plusieurs générations séparées par des résultats d’outils |

Deux définitions de **step** coexistent également :

| Sens de « step » | Ce qui est compté |
|---|---|
| Action step | chaque appel d’outil |
| Interaction step | chaque round génération → outil → observation |

Donc, ne conclus jamais le fonctionnement d’un système à partir du seul mot *multi-turn* ou *multi-step*.

## Le test à appliquer en lisant du code

Cherche une structure de ce type :

```python
while not done:
    model_output = generate(context)
    tool_calls = parse(model_output)
    tool_results = execute(tool_calls)
    context += model_output + tool_results
```

Si la boucle revient à `generate(context)` après avoir ajouté `tool_results`, le rollout possède plusieurs générations dépendantes des observations : c’est un **multi-step agent loop**.

En revanche :

```python
model_output = generate(prompt)
tool_calls = parse(model_output)
reward = verify(tool_calls)
```

correspond à une **single-generation approximation**, même si `model_output` contient plusieurs appels d’outils.

---

## Application à Workplace Assistant

La description la moins ambiguë est :

> **Une seule demande utilisateur, mais potentiellement plusieurs étapes modèle–outil.**

Le rollout complet peut être :

```text
demande utilisateur
→ génération
→ appel(s) d’outil
→ résultat(s)
→ nouvelle génération
→ ...
→ réponse finale
```

Il peut donc contenir :

- un seul **user turn** ;
- plusieurs **model turns/generations** ;
- plusieurs **tool calls** ;
- plusieurs **interaction steps**.

## Pourquoi le verifier paraît single-step

Le verifier reçoit à la fin la liste aplatie de tous les appels :

```python
[
    call_a,
    call_b,
    call_c,
]
```

Puis il les rejoue pour comparer l’état final prédit à l’état de référence.

Il ne conserve pas nécessairement les frontières entre générations. Par conséquent :

> Le verifier indique **comment les appels sont évalués**, pas **comment ils ont été générés**.

La même liste peut provenir :

```text
Génération 1 : [A, B, C]
```

ou :

```text
Génération 1 : A
→ résultat A
→ Génération 2 : B
→ résultat B
→ Génération 3 : C
```

## Application à VERL

### Version initiale simplifiée

```text
une génération
→ parser tous les appels
→ les rejouer
→ comparer les états
→ reward
```

Nom précis :

> **Single-generation / single-model-turn Workplace baseline**

Cette version permet de tester le dataset, le parser, les outils et le verifier. Elle ne permet pas au modèle d’utiliser un résultat d’outil pour décider de l’appel suivant.

### Version agentique complète

```text
génération
→ exécution des outils
→ ajout des résultats au contexte
→ nouvelle génération
→ ...
→ reward final
```

Elle nécessite un **agent loop**. Les tokens générés par le modèle sont entraînables, tandis que les résultats d’outils servent seulement de contexte.

MCQA et Structured Outputs peuvent rester single-generation dans le même entraînement multi-domain ; seul le chemin de rollout de Workplace doit utiliser la boucle agentique.

---

## Phrase à retenir

> **Un message utilisateur ne signifie pas forcément une seule génération du modèle, et plusieurs appels d’outils ne signifient pas forcément plusieurs générations.**

## Mini-quiz

1. Le modèle génère trois appels d’outils dans une seule réponse. Combien de générations ?
2. Le modèle appelle un outil, reçoit son résultat, puis génère une nouvelle réponse. Single-step ou multi-step ?
3. Un verifier rejoue une liste aplatie d’appels. Peut-on en déduire combien de générations les ont produits ?
4. Quelle question faut-il poser pour choisir entre le rollout single-generation et un agent loop dans VERL ?

### Réponses

1. Une génération, avec trois appels.
2. Multi-step au niveau modèle–environnement.
3. Non.
4. « Le LLM doit-il être appelé de nouveau après avoir reçu le résultat d’un outil ? »# Fiche de révision — Single-turn, multi-turn, single-step et multi-step

## La règle la plus importante

Les mots **turn** et **step** sont employés différemment selon les articles et les frameworks.

Pour éviter toute confusion, compte toujours séparément :

1. les **messages utilisateur** ;
2. les **invocations du LLM** (générations) ;
3. les **appels d’outils** (actions) ;
4. les **interactions modèle–environnement** ;
5. l’**épisode complet**.

La question opérationnelle la plus importante est :

> **Le LLM est-il appelé une nouvelle fois après avoir reçu le résultat d’un outil ?**

- **Non** → une seule génération du modèle.
- **Oui** → trajectoire agentique avec plusieurs générations/étapes.

---

## Les unités à distinguer

| Unité | Définition |
|---|---|
| **User turn** | Un message envoyé par l’utilisateur |
| **Model turn / generation** | Une invocation du LLM |
| **Tool call / action** | Un appel d’outil demandé par le LLM |
| **Tool observation** | Le résultat renvoyé par l’outil au modèle |
| **Interaction step** | Généralement : une génération du modèle suivie de l’exécution de ses actions |
| **Episode / trajectory** | Toute la résolution d’une tâche, du prompt initial à la fin |

## Cas 1 — Réponse classique

```text
Utilisateur
    ↓
Génération 1 : réponse finale
```

- 1 message utilisateur ;
- 1 génération du LLM ;
- 0 appel d’outil ;
- épisode single-step ;
- conversation single-turn.

## Cas 2 — Plusieurs appels d’outils générés ensemble

```text
Utilisateur
    ↓
Génération 1 : [appel A, appel B, appel C]
    ↓
Exécution des trois appels
```

- 1 message utilisateur ;
- 1 génération du LLM ;
- 3 appels d’outils ;
- 1 round modèle–environnement ;
- **single-model-turn, multi-call**.

Les trois appels peuvent être générés dans la même réponse, par exemple :

```xml
<tool_call>{"name": "tool_a", "arguments": {...}}</tool_call>
<tool_call>{"name": "tool_b", "arguments": {...}}</tool_call>
```

Ce cas ne devient pas multi-turn uniquement parce qu’il contient plusieurs appels.

Certains articles peuvent néanmoins compter chaque action comme un « step » et appeler ce cas *multi-step*. C’est pour cela que **single-turn, multi-call** est une description plus précise.

## Cas 3 — Le modèle reçoit les résultats et génère de nouveau

```text
Utilisateur
    ↓
Génération 1 : rechercher l’email de Carlos
    ↓
Outil : carlos@example.com
    ↓
Génération 2 : envoyer un email à carlos@example.com
    ↓
Outil : email envoyé
    ↓
Génération 3 : réponse finale
```

- 1 message utilisateur ;
- plusieurs générations du LLM ;
- plusieurs interactions modèle–environnement ;
- trajectoire **multi-step** ;
- souvent appelée **multi-turn tool use** ou **multi-step agentic rollout**.

Elle reste cependant *single-turn* si l’auteur réserve le mot *turn* aux messages humains, car l’utilisateur n’a envoyé qu’un seul message.

## Cas 4 — Conversation humaine multi-turn

```text
Utilisateur → Assistant → Utilisateur → Assistant
```

- plusieurs messages utilisateur ;
- plusieurs tours de conversation ;
- **multi-turn conversation** au sens humain.

Ce n’est pas la même chose qu’un agent qui effectue plusieurs appels d’outils après une seule demande.

---

## Pourquoi la terminologie paraît contradictoire

Deux définitions de **turn** coexistent :

| Sens de « turn » | « Multi-turn » signifie… |
|---|---|
| Conversation humaine | plusieurs messages utilisateur |
| Interaction agentique | plusieurs générations séparées par des résultats d’outils |

Deux définitions de **step** coexistent également :

| Sens de « step » | Ce qui est compté |
|---|---|
| Action step | chaque appel d’outil |
| Interaction step | chaque round génération → outil → observation |

Donc, ne conclus jamais le fonctionnement d’un système à partir du seul mot *multi-turn* ou *multi-step*.

## Le test à appliquer en lisant du code

Cherche une structure de ce type :

```python
while not done:
    model_output = generate(context)
    tool_calls = parse(model_output)
    tool_results = execute(tool_calls)
    context += model_output + tool_results
```

Si la boucle revient à `generate(context)` après avoir ajouté `tool_results`, le rollout possède plusieurs générations dépendantes des observations : c’est un **multi-step agent loop**.

En revanche :

```python
model_output = generate(prompt)
tool_calls = parse(model_output)
reward = verify(tool_calls)
```

correspond à une **single-generation approximation**, même si `model_output` contient plusieurs appels d’outils.

---

## Application à Workplace Assistant

La description la moins ambiguë est :

> **Une seule demande utilisateur, mais potentiellement plusieurs étapes modèle–outil.**

Le rollout complet peut être :

```text
demande utilisateur
→ génération
→ appel(s) d’outil
→ résultat(s)
→ nouvelle génération
→ ...
→ réponse finale
```

Il peut donc contenir :

- un seul **user turn** ;
- plusieurs **model turns/generations** ;
- plusieurs **tool calls** ;
- plusieurs **interaction steps**.

## Pourquoi le verifier paraît single-step

Le verifier reçoit à la fin la liste aplatie de tous les appels :

```python
[
    call_a,
    call_b,
    call_c,
]
```

Puis il les rejoue pour comparer l’état final prédit à l’état de référence.

Il ne conserve pas nécessairement les frontières entre générations. Par conséquent :

> Le verifier indique **comment les appels sont évalués**, pas **comment ils ont été générés**.

La même liste peut provenir :

```text
Génération 1 : [A, B, C]
```

ou :

```text
Génération 1 : A
→ résultat A
→ Génération 2 : B
→ résultat B
→ Génération 3 : C
```

## Application à VERL

### Version initiale simplifiée

```text
une génération
→ parser tous les appels
→ les rejouer
→ comparer les états
→ reward
```

Nom précis :

> **Single-generation / single-model-turn Workplace baseline**

Cette version permet de tester le dataset, le parser, les outils et le verifier. Elle ne permet pas au modèle d’utiliser un résultat d’outil pour décider de l’appel suivant.

### Version agentique complète

```text
génération
→ exécution des outils
→ ajout des résultats au contexte
→ nouvelle génération
→ ...
→ reward final
```

Elle nécessite un **agent loop**. Les tokens générés par le modèle sont entraînables, tandis que les résultats d’outils servent seulement de contexte.

MCQA et Structured Outputs peuvent rester single-generation dans le même entraînement multi-domain ; seul le chemin de rollout de Workplace doit utiliser la boucle agentique.

---

## Phrase à retenir

> **Un message utilisateur ne signifie pas forcément une seule génération du modèle, et plusieurs appels d’outils ne signifient pas forcément plusieurs générations.**

## Mini-quiz

1. Le modèle génère trois appels d’outils dans une seule réponse. Combien de générations ?
2. Le modèle appelle un outil, reçoit son résultat, puis génère une nouvelle réponse. Single-step ou multi-step ?
3. Un verifier rejoue une liste aplatie d’appels. Peut-on en déduire combien de générations les ont produits ?
4. Quelle question faut-il poser pour choisir entre le rollout single-generation et un agent loop dans VERL ?

### Réponses

1. Une génération, avec trois appels.
2. Multi-step au niveau modèle–environnement.
3. Non.
4. « Le LLM doit-il être appelé de nouveau après avoir reçu le résultat d’un outil ? »