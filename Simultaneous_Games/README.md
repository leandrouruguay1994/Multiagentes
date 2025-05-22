# README 

Este proyecto se realizó para presentar a la materia de Sistemas Multiagentes y está compuesto por tres carpetetas que contiene la estructura de nuestro código, como también una carpeta de resultados de nuestras notebooks que se encuentran en la raiz del proyecto.

Pasaremos a describir cada una de estas carpetas con el contenido de esta:

# Agentes
Contiene los agentes que interacturan con los dieferentes games (ambientes)

Cada agente tiene su propio archivo de python y estos son:

* ficticiousplay.py -> Agente basado en Fictitious Play
* iql_agent.py -> Agente de aprendizaje independiente (IQL)
* jal_am_agent.py -> Agente con aprendizaje conjunto (JAL-AM)
* random_agent.py -> Agente aleatorio
* regretmatching.py -> Agente basado en Regret Matching

# base
Define las abstracciones básicas del entorno multiagente:

* agent.py -> Clase base para todos los agentes
* game.py -> Clase base para la implementación de entornos, utilizando SimultaneousGame, entre otros

# games
Contiene la implementación de los distintos juegos o entornos en los que los agentes interactúan:

* blotto.py -> Juego de Blotto (con configuraciones variables de soldados y campos)
* foraging.py -> Entorno estocástico basado en el juego Level-Based Foraging
* mp.py -> Matching Pennies
* rps.py -> Rock-Paper-Scissors

# notebook_exportss
Carpeta utilizada para guardar resultados exportados desde las notebooks.

# Archivo N-form_games
Archivo para realizar los analisis en nuestros amibentes de MP, RPS y Blotto, utilizando los agentes: ficticiouosplay, regretmatching y random_agent.

# Archivo Stochastic_Games
Archivo para realizar los analisis en nuestros amibente de foraging, utilizando los agentes: iql_agent y jal_am_agent.
