# 🛰️ Lunar Lander PPO - Agente Adaptativo

Este projeto implementa um agente de aprendizado por reforço profundo (Deep RL) utilizando o algoritmo **PPO (Proximal Policy Optimization)** para controlar uma nave no ambiente `LunarLander-v3`, com múltiplas etapas de dificuldade crescente. O agente aprende a realizar pousos suaves e eficientes por meio de um currículo adaptativo.

---

## 📂 Estrutura do Projeto

- `treinamento.py`: Script de treinamento com curriculum learning e mudanças dinâmicas de dificuldade.
- `agente.py`: Script de avaliação e teste do modelo treinado com visualização, análise e gráficos.

---

## 🚀 Características

### 🎓 Aprendizado com Currículo Adaptativo
O script de treinamento aplica três fases:
1. **Fase Inicial**: Ambiente padrão com seed fixa.
2. **Fase de Generalização**: Ativa seed aleatória ao atingir recompensa média ≥ 200.
3. **Modo Difícil**: Introduz instabilidades físicas quando recompensa ≥ 240.
4. **Refino**: Penalidade por passos ativada ao atingir ≥ 260.

### 🧠 Arquitetura do Modelo

```python
ActorCriticPolicy(
  MlpExtractor:
    policy_net:  [8 → 64 → 64]
    value_net:   [8 → 64 → 64]
)
Parâmetros totais: 9.797

📊 Resultados
✅ Desempenho final (teste com 50 episódios)

    Taxa de sucesso: 94.0% (47/50)

    Recompensa média: 274.70 ± 51.71

    Recompensa máxima: 322.00

    Pousos suaves: 50/50

    Tempo médio por episódio: 5.54s

    Velocidade vertical média: 0.00

    Ângulo médio: 0.01 rad

Gráfico de desempenho salvo como: performance_graph.png
🧪 Treinamento Final

    Timesteps: 12.000.000

    Recompensa média (últimos 10): 288.83

    Recompensa avaliada: 262.77 ± 36.83

    Explained Variance: 0.998

    Entropy Loss: -0.228

    FPS: 28

▶️ Como usar
📦 Requisitos

    Python 3.10+

    stable-baselines3

    gymnasium

    matplotlib

    numpy

🏋️‍♂️ Treinamento

python treinamento.py

🎮 Avaliação do agente

python agente.py

👨‍💻 Autor

Raul Campos Nascimento
📧 rautopiaa@gmail.com
