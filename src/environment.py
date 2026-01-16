import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DatasetACEnv(gym.Env):
    """
    Env sencillo que recorre filas de un DataFrame con columnas:
    [temp_interna, num_personas, ubic_rel_onehot..., opinion_onehot..., horario_bool, temp_externa, ac_state]
    Acción discreta: 0=apagar AC, 1=encender AC
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, comfort_temp_center=24.5,
                 temp_min=18.0, temp_max=38.0, max_people=20, cooling_rate = 0.12, warming_rate = 0.24, cooling_rate_max = 0.14, cooling_rate_min = 0.05, max_steps_per_episode = 300): ## cooling_rate = 0.12 °C por step , warming_rate ≈ 2 * cooling_rate

        super().__init__()
        self.df = df.reset_index(drop=True)
        self.N = len(df)
        self.idx = 0

        # Acción discreta: 5 acciones posibles
        self.ACTIONS = {
            0: "Prender ventilación",
            1: "Bajar temperatura",
            2: "No hacer nada",
            3: "Subir temperatura",
            4: "Apagar ventilación"
        }
        self.action_space = spaces.Discrete(len(self.ACTIONS))



        # Construir observation_space: concatenación de features (usar normalización en build_state)
        # Para ejemplo usaremos un vector de tamaño variable; definimos un rango amplio
        obs_len = 1 + 1 + 3 + 3 + 3 + 1 + 1  # ajusta según columnas reales: temp,num,ubic(3),opinion(3),horario(3 para Dqn, ya que se hizo un cambio en los features de machine learning),temp_ext,ac_state
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_len,), dtype=np.float32)

        # parámetros de recompensa
        self.comfort_center = comfort_temp_center
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.max_people = max_people
        self.cooling_rate = cooling_rate
        self.warming_rate = warming_rate
        self.cooling_rate_max = cooling_rate_max
        self.cooling_rate_min = cooling_rate_min

        # pesos
        self.alpha = 0.5  # penalidad por no estar cómodo
        self.beta = 0.1   # costo por tener AC on
        self.gamma = 0.6  # recompensa por respetar regla de apagar cuando nadie y no hay clase
        self.gamma_big = 1.0

        # ==============================
        # ESTADO INTERNO PERSISTENTE HVAC
        # ==============================
        self.current_temp = None
        self.current_ac_state = None


        # ==============================
        # ÍNDICES AGRUPADOS POR DÍA
        # ==============================
        self.day_indices = (
            self.df.groupby("Fecha")
            .apply(lambda x: x.index.to_list())
            .to_list()
        )

        self.current_day = None
        self.day_rows = None
        self.step_in_day = 0


        self.max_steps_per_episode = max_steps_per_episode
        self.step_counter = 0


    def sample_thermal_opinion(self, temp, n_personas, prev_opinion=None):
        if n_personas == 0:
            return 1  # neutro fijo

        delta = temp - self.comfort_center

        if abs(delta) < 1.0:
            probs = [0.1, 0.8, 0.1]
        elif delta < 0:
            probs = [0.7, 0.2, 0.1]
        else:
            probs = [0.1, 0.2, 0.7]

        # histéresis humana
        if prev_opinion is not None and np.random.rand() < 0.3:
            return prev_opinion

        return np.random.choice([0, 1, 2], p=probs)



    def build_state(self, row):
        # row es una Series del DataFrame
        # normaliza
        t = self.current_temp # Default if NaN
        t_norm = (t - self.temp_min) / (self.temp_max - self.temp_min)  # 0..1
        #t = float(row["temp_interna"]) if pd.notna(row["temp_interna"]) else self.comfort_center # Default if NaN

        people = float(row["n_personas"]) if pd.notna(row["n_personas"]) else 0.0
        people_norm = people / self.max_people  # 0..1 (clamp)
        people_norm = np.clip(people_norm, 0.0, 1.0)

        # # asumir columnas:
        # # Ubicación relativa como lista one-hot en columnas ['ubic_cerca','ubic_lejos','ubic_dispersos']
        # ubic = np.array([row.get("ubic_cerca", 0.0), row.get("ubic_lejos", 0.0), row.get("ubic_dispersos", 0.0)], dtype=float)

        # # Opinión térmica one-hot ['op_frio','op_comodo','op_calor']
        # opinion = np.array([row.get("op_frio",0.0), row.get("op_comodo",0.0), row.get("op_calor",0.0)], dtype=float)

        # One-hot encoding para 'ubicacion_relativa' (0: dispersas, 1: cerca de ventilación, 2: lejos)
        ubicacion_relativa_val = row["ubicacion_relativa"]
        ubic = np.zeros(3, dtype=float)
        if pd.notna(ubicacion_relativa_val):
            ubicacion_relativa = int(ubicacion_relativa_val)
            if 0 <= ubicacion_relativa < 3: # Ensure index is valid
                ubic[ubicacion_relativa] = 1.0  # activa solo la categoría correspondiente

        # One-hot encoding para 'thermal_opinion' (0: fría, 1: neutra, 2: calurosa)
        thermal_opinion_val = row["thermal_opinion"]
        opinion = np.zeros(3, dtype=float)
        if pd.notna(thermal_opinion_val):
            thermal_opinion = int(thermal_opinion_val)
            if 0 <= thermal_opinion < 3: # Ensure index is valid
                opinion[thermal_opinion] = 1.0


        #horario = 1.0 if bool(row["clases_a_continuacion"]) else 0.0

        # One-hot encoding para 'clases_a_continuacion' (0: nada (laboratorio cerrado y no hay clases), 1: laboratorio abierto, 2: Clases practicas)
        clases_a_continuacion_val = row["clases_a_continuacion"]
        horario = np.zeros(3, dtype=float)
        if pd.notna(clases_a_continuacion_val):
            clases_a_continuacion = int(clases_a_continuacion_val)
            if 0 <= clases_a_continuacion < 3: # Ensure index is valid
                horario[clases_a_continuacion] = 1.0


        temp_ext = float(row["temp_externa"]) if pd.notna(row["temp_externa"]) else self.comfort_center # Default if NaN
        temp_ext_norm = (temp_ext - self.temp_min) / (self.temp_max - self.temp_min)
        #ac_state = 1.0 if bool(row["estado del aire"]) else 0.0
        ac_state = 1.0 if self.current_ac_state else 0.0

        vec = np.concatenate([
            [t_norm],          # temp_interna
            [temp_ext_norm],   # temp_externa
            [ac_state],         # estado del aire
            horario,         # clases_a_continuacion
            [people_norm],     # n_personas
            ubic,              # ubicacion_relativa (one-hot)
            opinion           # thermal_opinion (one-hot)
        ]).astype(np.float32)

        return vec

    def reset(self, seed=None, options=None, Trainment=True):
        super().reset(seed=seed)

        self.step_counter = 0

        if Trainment:
            # ==============================
            # ENTRENAMIENTO — INICIO ALEATORIO
            # ==============================
            self.idx = np.random.randint(0, self.N - self.max_steps_per_episode)
            self.day_rows = None
            self.step_in_day = 0

        else:
            # ==============================
            # EVALUACIÓN — EPISODIO = DÍA
            # ==============================
            self.current_day = np.random.randint(len(self.day_indices))
            self.day_rows = self.day_indices[self.current_day]

            self.step_in_day = 0
            self.idx = self.day_rows[self.step_in_day]

        # ==============================
        # ESTADO INTERNO PERSISTENTE
        # ==============================
        self.current_temp = float(self.df.loc[self.idx]["temp_interna"])
        self.current_ac_state = bool(self.df.loc[self.idx]["estado del aire"])

        state = self.build_state(self.df.loc[self.idx])
        info = {
            "index": self.idx,
            "training": Trainment,
            "day": self.current_day if not Trainment else None
        }

        return state, info



    def step(self, action):
        row = self.df.loc[self.idx].copy()
        row["estado del aire"] = self.current_ac_state
        row["temp_interna"] = self.current_temp

        temp_ext = float(row["temp_externa"]) if pd.notna(row["temp_externa"]) else self.comfort_center

        # === Interpretación de acción ===

        ac_state = self.current_ac_state
        temp_setpoint = self.current_temp
        #print("previous ac_state " + str(ac_state))                   ####PRINT
        #print("previous temp_setpoint " + str(temp_setpoint))         ####PRINT

        # ==============================
        # VALIDACIÓN DE ACCIÓN
        # ==============================
        valid_actions = (
            [0, 2] if not ac_state else [1, 2, 3, 4]
        )

        if action not in valid_actions:
            # fallback seguro
            action = 2  # No hacer nada
            invalid_action = True
        else:
            invalid_action = False


        if ac_state == False and action == 0:   # Prender ventilación
            ac_state = True
            temp_setpoint -= self.cooling_rate

        elif ac_state == True and action == 1:  # Bajar temperatura
            delta_T = abs(temp_ext - temp_setpoint)
            delta_norm = np.clip(delta_T / 5.0, 0.0, 1.0)

            cooling_rate = (
                self.cooling_rate_min
                + delta_norm * (self.cooling_rate_max - self.cooling_rate_min)
            )

            temp_setpoint -= cooling_rate

        elif action == 2:  # No hacer nada
            pass

        elif ac_state == True and action == 3:  # Subir temperatura
            delta_T = abs(temp_ext - temp_setpoint)
            delta_norm = np.clip(delta_T / 5.0, 0.0, 1.0)

            # transición progresiva hacia warming_rate
            relax_rate = delta_norm * self.warming_rate

            temp_setpoint += relax_rate


        elif ac_state == True and action == 4:  # Apagar ventilación
            ac_state = False
            # Calentamiento más rápido hacia temperatura externa
            temp_setpoint += self.warming_rate * np.sign(temp_ext - temp_setpoint)

        else:
            raise RuntimeError(
                f"Acción inválida {action} con AC={ac_state}"
            )



        # Guardar estado persistente
        self.current_temp = temp_setpoint
        self.current_ac_state = ac_state

        # ==============================
        # AVANCE DEL ÍNDICE Y DONE
        # ==============================

        self.step_counter += 1

        if self.day_rows is not None:
            # ===== MODO EVALUACIÓN: EPISODIO = DÍA =====
            self.step_in_day += 1
            done = self.step_in_day >= len(self.day_rows) - 1

            if not done:
                self.idx = self.day_rows[self.step_in_day]
            else:
                self.idx = self.day_rows[-1]

        else:
            # ===== MODO ENTRENAMIENTO =====
            done = self.step_counter >= self.max_steps_per_episode

            if not done:
                self.idx += 1
            else:
                self.idx = min(self.idx + 1, self.N - 1)

        next_row = self.df.loc[self.idx].copy()


        prev_opinion = row.get("thermal_opinion", 1)

        thermal_opinion = self.sample_thermal_opinion(
            temp=self.current_temp,
            n_personas=next_row["n_personas"],
            prev_opinion=prev_opinion
        )

        next_row["thermal_opinion"] = thermal_opinion


        # for reward calc, usamos la opinion real de next_row (si la tienes),
        # o inferimos confort desde temperatura


        # Calcular RECOMPENSA:
        # 1) COMFORT: preferir opinion == comodo (col 'op_comodo')
        # op_comodo = float(next_row.get("op_comodo", 0.0))
        # # si tienes opinión real: r_comfort = +1*op_comodo - 0.5*(1-op_comodo)
        # r_comfort = 1.0 * op_comodo - 0.5 * (1.0 - op_comodo)

        opinion = int(next_row["thermal_opinion"])


        num_people = int(next_row["n_personas"]) if pd.notna(next_row["n_personas"]) else 0

        if num_people == 0:
            # Sin personas → no existe confort humano
            r_comfort = 0.0
        else:
            # Penaliza desviación respecto al valor neutro (1)
            opinion = int(next_row["thermal_opinion"])
            r_comfort = -abs(opinion - 1.0)
            r_comfort *= self.alpha



        # 2) energy
        r_energy = - self.beta * (1.0 if ac_state else 0.0)

        # Penalización si cambia el estado del AC demasiado seguido (evita oscilaciones)
        if action in [0, 4]:
            r_switch = -0.1
        else:
            r_switch = 0.0

        # 3) idle-rule: si no hay gente y no hay clase -> AC==False

        horario = bool(next_row["clases_a_continuacion"]) if pd.notna(next_row["clases_a_continuacion"]) else False
        if (num_people == 0.0) and (not horario):
            if not ac_state:
                r_idle = self.gamma
            else:
                r_idle = -self.gamma_big
        else:
            r_idle = 0.0

        # ==============================
        # COMPONENTES DE LA RECOMPENSA
        # ==============================

        reward_components = {
            "r_comfort": r_comfort,
            "r_energy": r_energy,
            "r_switch": r_switch,
            "r_idle": r_idle
        }


        reward = r_comfort + r_energy  + r_switch + r_idle

        # construir estado de salida: usamos next_row pero con ac overwrite si quieres
        next_row["estado del aire"] = ac_state
        #print("new ac_state " + str(ac_state))           ####PRINT
        next_row["temp_interna"] = temp_setpoint
        #print("new temp_setpoint " + str(temp_setpoint)) ####PRINT
        obs = self.build_state(next_row)

        info = {
            "index": self.idx,
            "reward_components": reward_components
        }


        #print(
        #      f"STEP {self.step_counter} | "
        #      f"action={action} | "
        #      f"T={self.current_temp:.2f} | "
        #      f"AC={self.current_ac_state}"
        #  )

        return obs, float(reward), done, False, info  # gymnasium: terminated, truncated (we use terminated only)

    def render(self, mode="human", action=None):
        """
        Renderiza el estado actual del entorno.

        Parameters
        ----------
        mode : str
            - "human": imprime el estado por consola
            - "get": devuelve un diccionario estructurado del estado
        action : int, optional
            Acción tomada por el agente (solo requerido en mode="get")
        """

        row = self.df.loc[self.idx]

        if mode == "human":
            print(
                f"idx={self.idx} | "
                f"temp_interna={self.current_temp:.2f} | "
                f"temp_externa={row['temp_externa']:.2f} | "
                f"AC={'ON' if self.current_ac_state else 'OFF'} | "
                f"clases={row['clases_a_continuacion']} | "
                f"n_personas={row['n_personas']} | "
                f"ubicacion={row['ubicacion_relativa']} | "
                f"thermal_opinion={row['thermal_opinion']}"
            )

        elif mode == "get":
            if action is None:
                raise ValueError("mode='get' requiere que se pase la acción tomada")

            return {
                "idx": self.idx,
                "temp_in": float(self.current_temp),
                "temp_out": float(row["temp_externa"]),
                "ac": bool(self.current_ac_state),
                "clases": int(row["clases_a_continuacion"]),
                "thermal": int(row["thermal_opinion"]),
                "n_personas": int(row["n_personas"]),
                "ubicacion": int(row["ubicacion_relativa"]),
                "action": int(action)
            }

        else:
            raise ValueError(f"Modo de render no soportado: {mode}")
