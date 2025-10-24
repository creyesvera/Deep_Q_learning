import pandas as pd
import numpy as np
#from q_learning import ACTIONS, ThermalEnv, q_learning_thermal
from src.environment import ACTIONS, ThermalEnv
from src.agent import q_learning_thermal



def train_q_learning(
    df_features,
    n_episodes=10000,
    max_moves=10,
    initial_eps=1.0,
    final_eps=0.05,
    decay_rate=0.99,
    alpha=0.5,
    gamma=0.9
):
    """
    Prepares df_features, initializes Q-table, and trains Q-learning. Returns policy and Q-table.
    All Q-learning hyperparameters are configurable.
    """
    # Rename columns if needed to match expected names
    col_map = {
        'n_personas': 'n_people',
        'ubicacion': 'location',
        'opinion_termica': 'thermal_opinion',
        'estado_aire': 'estado del aire',
        'temp_interna_discretizada': 'temp_interna_discretizada',
        'temp_externa_discretizada': 'temp_externa_discretizada',
        'clases a continuación': 'clases a continuación'
    }
    df_features = df_features.rename(columns=col_map)
    df_features['state'] = df_features[[
        'temp_interna_discretizada',
        'n_people',
        'location',
        'thermal_opinion',
        'clases a continuación',
        'temp_externa_discretizada',
        'estado del aire'
    ]].apply(tuple, axis=1)
    #print(df_features.head())
    unique_states = df_features['state'].unique()
    n_actions = len(ACTIONS)
    initial_q_table = {
        state: [0.0 for _ in range(n_actions)]
        for state in unique_states
    }
    #print(initial_q_table)
    rng = np.random.default_rng(seed=42)
    env = ThermalEnv()
    pi, q_table, q_tables_by_episode, rewards, td_errors_per_episode = q_learning_thermal(
        rng, env,
        n_episodes=n_episodes,
        max_moves=max_moves,
        initial_eps=initial_eps,
        final_eps=final_eps,
        decay_rate=decay_rate,
        alpha=alpha,
        gamma=gamma,
        q_table=initial_q_table
    )
    print("✅ Política aprendida (estado → acción óptima):\n")
    for s, a in list(pi.items())[:20]:  # muestra solo los primeros 20
        print(f"{s} → {ACTIONS[a]}")


    data = []
    for state, action in pi.items():
        data.append({
            'temp_interna_discretizada': state[0],
            'n_personas': state[1],
            'ubicacion': state[2],
            'opinion_termica': state[3],
            'clases a continuación': state[4],
            'temp_externa_discretizada': state[5],
            'estado_aire': state[6],
            'accion_recomendada': ACTIONS[action]  # traducir acción
        })

    pi_df = pd.DataFrame(data)



    return pi_df




def using_trained_model(df_features):
    """
    Loads a trained Q-table and predicts recommended actions for each row in df_features.
    Assumes ThermalEnv has a method encode_state_from_row(fila) that returns the state tuple.
    """
    


    import pandas as pd

    # Cargar política aprendida
    df_policy = pd.read_csv("learned_policy_with_interpretation.csv")


    # Convertir 'State' a tupla de enteros
    def parse_state(s):
        s = s.strip(" ()")
        return tuple(int(float(x)) for x in s.split(","))

    df_policy["State"] = df_policy["State"].apply(parse_state)

    # Crear diccionario de mapeo
    state_to_action = dict(zip(df_policy["State"], df_policy["Optimal Action"]))

    # Crear entorno
    env = ThermalEnv()

    # Función para obtener acción recomendada
    def get_action(fila):
        try:
            # Codificamos estado con las 7 variables en el orden correcto
            state = (
                int(fila["temp_interna_discretizada"]),
                int(fila["temp_externa_discretizada"]),
                int(fila["estado_aire"]),
                int(fila["n_personas"]),
                int(fila["ubicacion"]),
                int(fila["opinion_termica"]),
                int(fila["clases a continuación"])
            )
            # Buscar acción en diccionario
            return state_to_action.get(state, f"⚠️ Estado {state} no encontrado en política")
        except Exception as e:
            return f"❌ Error: {e}"

    # Aplicamos al df_features
    df_features["accion_recomendada"] = df_features.apply(get_action, axis=1)

    # Mostrar ejemplo
    print(df_features[["Fecha", "Hora", "accion_recomendada"]].head(10))

    # Verificar cobertura: cuántos estados tienen acción definida
    total = len(df_features)
    definidos = df_features["accion_recomendada"].apply(lambda x: not str(x).startswith("⚠️")).sum()
    print(f"Cobertura de la política: {definidos}/{total} ({definidos/total:.1%})")


    return df_features


