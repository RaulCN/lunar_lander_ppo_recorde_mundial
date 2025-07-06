# treinamento_melhorado.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os

# Flags globais
USE_RANDOM_SEED = False
HARD_DIFFICULTY = False
STEP_PENALTY_ON = False

# Wrapper para alternar seed adaptativa
class LunarLanderAdaptiveWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        seed = np.random.randint(0, 1e6) if USE_RANDOM_SEED else 42
        return self.env.reset(seed=seed, **kwargs)

# Wrapper para modo mais difícil
class HarderLanderWrapper(gym.Wrapper):
    def __init__(self, env, dificuldade_extra=False):
        super().__init__(env)
        self.dificuldade_extra = dificuldade_extra

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.dificuldade_extra:
            obs[0] += np.random.uniform(-0.1, 0.1)
            obs[1] += np.random.uniform(-0.1, 0.1)
            obs[2] += np.random.uniform(-0.2, 0.2)
            obs[3] += np.random.uniform(-0.2, 0.2)
            obs[4] += np.random.uniform(-0.05, 0.05)

        return obs, info

# Wrapper para penalidade por step
class StepPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalidade=0.1):
        super().__init__(env)
        self.penalidade = penalidade

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward -= self.penalidade
        return obs, reward, done, truncated, info

# Função para criar ambiente

def make_adaptive_env():
    env = gym.make("LunarLander-v3")
    env = LunarLanderAdaptiveWrapper(env)
    env = HarderLanderWrapper(env, dificuldade_extra=HARD_DIFFICULTY)
    if globals().get("STEP_PENALTY_ON", False):
        env = StepPenaltyWrapper(env, penalidade=0.1)
    return env

# Callback com curriculum e modos adaptativos
class CurriculumCallback(BaseCallback):
    def __init__(self, threshold=200, hard_threshold=240, refine_threshold=260, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.hard_threshold = hard_threshold
        self.refine_threshold = refine_threshold
        self.check_freq = check_freq
        self.seed_switched = False
        self.hard_mode = False
        self.refine_mode = False
        self.episode_rewards = []

    def _on_step(self) -> bool:
        global USE_RANDOM_SEED, HARD_DIFFICULTY, STEP_PENALTY_ON

        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_reward = self.locals['infos'][0]['episode']['r']
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)

        if (self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 10):
            mean_reward = np.mean(self.episode_rewards[-10:])
            if self.verbose:
                print(f"\nStep {self.n_calls}: Média de recompensa (10 últimos): {mean_reward:.2f}")

            if mean_reward >= self.threshold and not self.seed_switched:
                USE_RANDOM_SEED = True
                self.seed_switched = True
                print("🎯 Seed aleatória ativada para generalização!")

            if mean_reward >= self.hard_threshold and not self.hard_mode:
                HARD_DIFFICULTY = True
                self.hard_mode = True
                print("🔥 Modo difícil ativado! Física mais instável no ambiente!")

            if mean_reward >= self.refine_threshold and not self.refine_mode:
                STEP_PENALTY_ON = True
                self.refine_mode = True
                print("⏱️ Ativando penalidade por passo para refinar eficiência do pouso!")

        return True

# Criar ambientes
print("Criando ambientes...")
train_env = Monitor(make_adaptive_env())
eval_env = Monitor(gym.make("LunarLander-v3"))

model_path = "ppo_lunar_adaptativo.zip"

# Carrega ou cria modelo
if os.path.exists(model_path):
    print("📂 Modelo encontrado. Carregando...")
    model = PPO.load(model_path, env=train_env)
else:
    print("🆕 Criando novo modelo PPO...")
    model = PPO("MlpPolicy", train_env,
                n_steps=2048,
                batch_size=64,
                gae_lambda=0.98,
                gamma=0.999,
                learning_rate=2.5e-4,
                ent_coef=0.01,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./logs/")

if not hasattr(model, 'meta_info'):
    model.meta_info = {
        "criador": "Raul Campos Nascimento",
        "algoritmo": "PPO",
        "ambiente": "LunarLander-v3",
        "versao": "1.4-curriculum+hard+refino",
        "seed_adaptativa": ">=200",
        "dificuldade_extra": ">=240",
        "penalidade_step": ">=260"
    }

os.makedirs("./melhor_modelo/", exist_ok=True)
os.makedirs("./logs/", exist_ok=True)

curriculum_cb = CurriculumCallback(threshold=200, hard_threshold=240, refine_threshold=260, check_freq=5000, verbose=1)
eval_cb = EvalCallback(eval_env,
                       best_model_save_path="./melhor_modelo/",
                       log_path="./logs/",
                       eval_freq=20000,
                       n_eval_episodes=10,
                       deterministic=True,
                       render=False)

print("\n🚀 Iniciando treinamento...")
print("📊 Parâmetros:")
print(f"   - Total de timesteps: 12.000.000")
print(f"   - Thresholds: Seed={curriculum_cb.threshold}, Dificuldade={curriculum_cb.hard_threshold}, Refino={curriculum_cb.refine_threshold}")
print(f"   - Verificação a cada: {curriculum_cb.check_freq} steps")
print(f"   - Avaliação a cada: 20.000 steps")

try:
    model.learn(total_timesteps=12_000_000,
                callback=CallbackList([curriculum_cb, eval_cb]),
                progress_bar=True)
    model.save(model_path)
    print("\n✅ Treinamento finalizado e modelo salvo!")
    print("📋 Metainformações:", model.meta_info)
except KeyboardInterrupt:
    print("\n⚠️  Treinamento interrompido pelo usuário")
    model.save(model_path)
    print("💾 Modelo salvo antes de encerrar")
except Exception as e:
    print(f"\n❌ Erro durante o treinamento: {e}")
finally:
    train_env.close()
    eval_env.close()
    print("🔚 Ambientes fechados")
