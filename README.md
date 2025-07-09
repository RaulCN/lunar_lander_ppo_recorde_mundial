# Lunar Lander PPO - Agente Adaptativo

Este projeto implementa um agente de aprendizado por reforço profundo (Deep Reinforcement Learning) utilizando o algoritmo **PPO (Proximal Policy Optimization)** para controlar uma nave no ambiente `LunarLander-v3`. O agente é treinado com múltiplas etapas de dificuldade crescente por meio de um currículo adaptativo, permitindo pousos suaves e eficientes em cenários cada vez mais desafiadores.

## Estrutura do Projeto

- `treinamento.py`: Script de treinamento com currículo adaptativo e progressão dinâmica de dificuldade.
- `agente.py`: Script de avaliação e teste do modelo treinado, com visualização, análise estatística e geração de gráficos.

## Características Técnicas

### Aprendizado com Currículo Adaptativo

O script de treinamento aplica quatro fases distintas:
1. **Fase Inicial**: Ambiente padrão com semente fixa.
2. **Generalização**: Ativação de sementes aleatórias ao atingir recompensa média ≥ 200.
3. **Modo Difícil**: Introdução de instabilidades físicas quando a média atinge ≥ 240.
4. **Refino**: Penalidade por número de passos ativada ao atingir média ≥ 260.

### Arquitetura do Modelo

```python
ActorCriticPolicy(
  MlpExtractor:
    policy_net:  [8 → 64 → 64]
    value_net:   [8 → 64 → 64]
)
# Total de parâmetros: 9.797
```

## Resultados

### Avaliação Final (50 episódios)

- **Taxa de sucesso:** 94.0% (47/50)
- **Recompensa média:** 274.70 ± 51.71
- **Recompensa máxima:** 322.00
- **Pousos suaves:** 50/50
- **Tempo médio por episódio:** 5.54s
- **Velocidade vertical média:** 0.00
- **Ângulo médio:** 0.01 rad

Gráfico de desempenho salvo como `performance_graph.png`.

### Dados do Treinamento Final

- **Timesteps totais:** 12.000.000
- **Recompensa média (últimos 10 episódios):** 288.83
- **Recompensa avaliada:** 262.77 ± 36.83
- **Explained Variance:** 0.998
- **Entropy Loss:** -0.228
- **FPS:** 28

## Execução

### Requisitos

- Python 3.10+
- stable-baselines3
- gymnasium
- matplotlib
- numpy

### Treinar o modelo

```bash
python treinamento.py
```

### Avaliar o agente

```bash
python agente.py
```

## Acesso ao Modelo

Os pesos treinados do agente estão disponíveis no Hugging Face:

**Hugging Face Hub:**  
https://huggingface.co/rautopia/ppo-lunar-lander-v3-max322

## Autor

**Raul Campos Nascimento**  
Email: rautopiaa@gmail.com
