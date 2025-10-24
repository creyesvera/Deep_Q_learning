# -*- coding: utf-8 -*-
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
                 temp_min=15.0, temp_max=38.0, max_people=20):
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
        obs_len = 1 + 1 + 3 + 3 + 1 + 1 + 1  # ajusta según columnas reales: temp,num,ubic(3),opinion(3),horario,temp_ext,ac_state
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_len,), dtype=np.float32)

        # parámetros de recompensa
        self.comfort_center = comfort_temp_center
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.max_people = max_people

        # pesos
        self.alpha = 0.5  # penalidad por no estar cómodo
        self.beta = 0.1   # costo por tener AC on
        self.gamma = 0.6  # recompensa por respetar regla de apagar cuando nadie y no hay clase
        self.gamma_big = 1.0

    def build_state(self, row):
        # row es una Series del DataFrame
        # normaliza temp
        t = float(row["temp_interna"])
        t_norm = (t - self.temp_min) / (self.temp_max - self.temp_min)  # 0..1

        people = float(row["n_people"])
        people_norm = people / self.max_people  # 0..1 (clamp)
        people_norm = np.clip(people_norm, 0.0, 1.0)

        # # asumir columnas:
        # # Ubicación relativa como lista one-hot en columnas ['ubic_cerca','ubic_lejos','ubic_dispersos']
        # ubic = np.array([row.get("ubic_cerca", 0.0), row.get("ubic_lejos", 0.0), row.get("ubic_dispersos", 0.0)], dtype=float)

        # # Opinión térmica one-hot ['op_frio','op_comodo','op_calor']
        # opinion = np.array([row.get("op_frio",0.0), row.get("op_comodo",0.0), row.get("op_calor",0.0)], dtype=float)

        # One-hot encoding para 'location' (0: dispersas, 1: cerca de ventilación, 2: lejos)
        location = int(row["location"])
        ubic = np.zeros(3, dtype=float)
        ubic[location] = 1.0  # activa solo la categoría correspondiente

        # One-hot encoding para 'thermal_opinion' (0: fría, 1: neutra, 2: calurosa)
        thermal_opinion = int(row["thermal_opinion"])
        opinion = np.zeros(3, dtype=float)
        opinion[thermal_opinion] = 1.0


        horario = 1.0 if bool(row["clases a continuación"]) else 0.0
        temp_ext = float(row["temp_externa"])
        temp_ext_norm = (temp_ext - self.temp_min) / (self.temp_max - self.temp_min)
        ac_state = 1.0 if bool(row["estado del aire"]) else 0.0

        vec = np.concatenate([
            [t_norm],          # temp_interna
            [temp_ext_norm],   # temp_externa
            [ac_state],         # estado del aire
            [horario],         # clases a continuación
            [people_norm],     # n_people
            ubic,              # location (one-hot)
            opinion           # thermal_opinion (one-hot)
        ]).astype(np.float32)

        return vec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # comenzar en 0 o aleatorio
        self.idx = 0  # o: self.np_random.integers(0, self.N-1)
        state = self.build_state(self.df.loc[self.idx])
        info = {"index": self.idx}
        return state, info

    def step(self, action):
        row = self.df.loc[self.idx].copy()

        # === Interpretación de acción ===

        ac_state = row["estado del aire"]  # estado previo (True/False)
        temp_setpoint = self.comfort_center  # si tienes una variable objetivo de temperatura


        if action == 0:   # Prender ventilación
            ac_state = True

        elif action == 1:  # Bajar temperatura
            temp_setpoint -= 1.0  # o 0.5 si quieres un ajuste más suave

        elif action == 2:  # No hacer nada
            pass

        elif action == 3:  # Subir temperatura
            temp_setpoint += 1.0

        elif action == 4:  # Apagar ventilación
            ac_state = False

        # Aplicar acción (0/1)
        # Interpretamos AC según acción (sobrescribe columna 'estado del aire' en estado simulado)
        # row = self.df.loc[self.idx].copy()
        # ac = bool(action)  # 0->False, 1->True

        # construir siguiente estado (aquí uso la siguiente fila real; podrías predecir con modelo)
        next_idx = self.idx + 1
        done = next_idx >= self.N
        done = False
        if next_idx >= self.N:
            done = True
            next_idx = self.N - 1

        next_row = self.df.loc[next_idx].copy()
        # for reward calc, usamos la opinion real de next_row (si la tienes),
        # o inferimos confort desde temperatura


        # Calcular RECOMPENSA:
        # 1) COMFORT: preferir opinion == comodo (col 'op_comodo')
        # op_comodo = float(next_row.get("op_comodo", 0.0))
        # # si tienes opinión real: r_comfort = +1*op_comodo - 0.5*(1-op_comodo)
        # r_comfort = 1.0 * op_comodo - 0.5 * (1.0 - op_comodo)

        opinion = int(next_row["thermal_opinion"])
        # Penaliza desviación respecto al valor neutro (1)
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
        num_people = int(next_row["n_people"])
        horario = bool(next_row["clases a continuación"])
        if (num_people == 0.0) and (not horario):
            if not ac_state:
                r_idle = self.gamma
            else:
                r_idle = -self.gamma_big
        else:
            r_idle = 0.0

        reward = r_comfort + r_energy  + r_switch + r_idle

        # construir estado de salida: usamos next_row pero con ac overwrite si quieres
        next_row["estado del aire"] = ac_state
        next_row["temp_interna"] = temp_setpoint
        obs = self.build_state(next_row)

        self.idx = next_idx
        info = {"index": self.idx}

        #print(f"Paso {self.idx}: temp={next_row['temp_interna']:.2f}, n_people={next_row['n_people']}, action={action}")


        return obs, float(reward), done, False, info  # gymnasium: terminated, truncated (we use terminated only)

    def render(self, mode="human"):
        r = self.df.loc[self.idx]
        # print(f"idx={self.idx} temp={r['Temperatura interna']} people={r['Número de personas']} horario={r['Horario de clase']} ac={r.get('Estado aire acondicionado')}")
        print(
              f"idx={self.idx} "
              f"temp_interna={r['temp_interna']} "
              f"temp_externa={r['temp_externa']} "
              f"estado_aire={r.get('estado_aire')} "
              f"horario_clase={r['horario_clase']} "
              f"n_personas={r['n_personas']} "
              f"location={r['location']} "
              f"thermal_opinion={r.get('thermal_opinion')}"
              )
