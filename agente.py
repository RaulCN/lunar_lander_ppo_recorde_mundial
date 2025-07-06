# agente_adaptativo.py
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import deque

class AdvancedLunarLanderTester:
    def __init__(self):
        self.model = self._load_model()
        self.env = gym.make("LunarLander-v3", render_mode="human")
        self.episode_data = []
        self.best_reward = -np.inf
        self.last_rewards = deque(maxlen=10)

    def _load_model(self) -> PPO:
        """Carrega o modelo mais recente disponível (prioriza o final, depois o melhor)"""
        base_path = os.path.dirname(__file__)
        model_paths = {
            'final': os.path.join(base_path, "ppo_lunar_adaptativo.zip"),
            'best': os.path.join(base_path, "melhor_modelo", "best_model.zip")
        }
        for name, path in model_paths.items():
            if os.path.exists(path):
                print(f"📂 Carregando modelo {name} de {path}...")
                model = PPO.load(path)
                self._print_model_info(model)
                return model
        raise FileNotFoundError("❌ Nenhum modelo encontrado. Execute treinamento.py primeiro.")

    def _print_model_info(self, model: PPO):
        print("✅ Modelo carregado com sucesso!")
        print(f"🔍 Arquitetura: {model.policy}")
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"⚙️ Parâmetros totais: {total_params}")
        if hasattr(model, 'meta_info'):
            print("\n📋 Metainformações do modelo:")
            for key, value in model.meta_info.items():
                print(f"   {key}: {value}")

    def _analyze_landing(self, obs: np.ndarray) -> Dict[str, float]:
        x, y, vx, vy, angle, vang, leg1, leg2 = obs
        return {
            'velocidade_vertical': abs(vy),
            'velocidade_horizontal': abs(vx),
            'angulo': abs(angle),
            'suavidade': (abs(vy) < 0.5 and abs(vx) < 0.5),
            'combustivel': obs[-1]
        }

    def run_test_episodes(self, num_episodes: int = 60, render_every: int = 5):
        print(f"\n🚀 Testando modelo por {num_episodes} episódios...")
        print(f"(Renderizando a cada {render_every} episódios)")
        successful_landings = 0
        landing_analysis = []
        start_time = time.time()

        for i in range(num_episodes):
            obs, _ = self.env.reset(seed=42 + i)
            done = truncated = False
            total_reward = 0
            steps = 0
            should_render = (i % render_every == 0)

            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                if should_render:
                    self.env.render()
                    time.sleep(0.01)
            
            # Análise de pouso
            landing_metrics = self._analyze_landing(obs)
            landing_analysis.append(landing_metrics)

            # Atualiza melhores e médias
            self.best_reward = max(self.best_reward, total_reward)
            self.last_rewards.append(total_reward)
            moving_avg = np.mean(self.last_rewards)

            # Classifica episódio
            if total_reward >= 200:
                successful_landings += 1
                status = "✅ SUCESSO"
            elif total_reward >= 100:
                status = "⚠️  OK"
            else:
                status = "❌ FALHA"

            # Tempo por episódio
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)

            # Impressão detalhada
            print(
                f"Episódio {i+1:2d}: Recompensa = {total_reward:6.2f} | Steps = {steps:3d} | "
                f"MovAvg(10) = {moving_avg:6.2f} | Best = {self.best_reward:6.2f} | {status}\n"
                f"    ⇨ VelV = {landing_metrics['velocidade_vertical']:.2f}, VelH = {landing_metrics['velocidade_horizontal']:.2f}, "
                f"Ângulo = {landing_metrics['angulo']:.2f}, Fuel = {landing_metrics['combustivel']:.2f}, "
                f"Tempo Médio = {avg_time:.2f}s/ep\n"
            )

            self.episode_data.append({
                'episode': i+1,
                'reward': total_reward,
                'steps': steps,
                **landing_metrics,
                'moving_avg': moving_avg,
                'best_reward': self.best_reward,
                'status': status,
                'avg_time': avg_time
            })

        self._generate_report(successful_landings, num_episodes, landing_analysis)

    def _generate_report(self, successes: int, total_episodes: int, landings: List[Dict]):
        rewards = [ep['reward'] for ep in self.episode_data]
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE PERFORMANCE AVANÇADO")
        print("="*60)
        print(f"Episódios testados: {total_episodes}")
        print(f"Taxa de sucesso: {successes}/{total_episodes} ({successes/total_episodes*100:.1f}%)")
        print(f"Recompensa média: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Máxima: {np.max(rewards):.2f} | Mínima: {np.min(rewards):.2f}")
        avg_times = [ep['avg_time'] for ep in self.episode_data]
        print(f"Tempo médio por episódio: {np.mean(avg_times):.2f}s")

        # Estatísticas de pouso
        vel_vs = [l['velocidade_vertical'] for l in landings]
        vel_hs = [l['velocidade_horizontal'] for l in landings]
        angles = [l['angulo'] for l in landings]
        smooths = sum(l['suavidade'] for l in landings)
        print(f"\n🔧 Análise de pousos ({len(landings)} renderizados):")
        print(f"Pousos suaves: {smooths}/{len(landings)}")
        print(f"Vel V média: {np.mean(vel_vs):.2f} | Vel H média: {np.mean(vel_hs):.2f}")
        print(f"Ângulo médio: {np.mean(angles):.2f} rad")

        # Gráfico de performance
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Recompensa')
        plt.plot([ep['moving_avg'] for ep in self.episode_data], label='MovAvg(10)')
        plt.title("Performance por Episódio")
        plt.xlabel("Episódio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.savefig("performance_graph.png")
        print("\n📈 Gráfico de performance salvo como 'performance_graph.png'")
        print("="*60)

    def close(self):
        self.env.close()
        print("🔚 Ambiente fechado")

if __name__ == "__main__":
    tester = None
    try:
        tester = AdvancedLunarLanderTester()
        tester.run_test_episodes(num_episodes=50, render_every=5)
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
    finally:
        if tester is not None:
            tester.close()
