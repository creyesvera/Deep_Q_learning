import pandas as pd
import numpy as np

def simulate_horario_df():
    horario = {
        'Hora': [
            '7:00-7:30', '7:30-8:00', '8:00-8:30', '8:30-9:00', '9:00-9:30',
            '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30', '11:30-12:00',
            '12:00-12:30', '12:30-13:00', '13:00-13:30', '13:30-14:00', '14:00-14:30',
            '14:30-15:00', '15:00-15:30', '15:30-16:00', '16:00-16:30', '16:30-17:00',
            '17:00-17:30', '17:30-18:00', '18:00-18:30'
        ],
        'Lunes': [
            'Vacío', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases',
            'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Clases', 'Clases', 'Clases',
            'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
        ],
        'Martes': [
            'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto', 'Clases',
            'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Vacío',
            'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto',
            'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
        ],
        'Miércoles': [
            'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto',
            'Laboratorio abierto', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases',
            'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Vacío', 'Vacío',
            'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
        ],
        'Jueves': [
            'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Laboratorio abierto', 'Laboratorio abierto',
            'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto', 'Clases',
            'Clases', 'Clases', 'Clases', 'Clases', 'Laboratorio abierto', 'Laboratorio abierto',
            'Laboratorio abierto', 'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío',
            'Vacío'
        ],
        'Viernes': [
            'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Clases', 'Clases', 'Clases', 'Clases',
            'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Vacío', 'Laboratorio abierto',
            'Laboratorio abierto', 'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío',
            'Vacío', 'Vacío'
        ]
    }
    df_horario = pd.DataFrame(horario)
    def get_activity_type(cell_content):
        if "Laboratorio abierto" in str(cell_content):
            return "Clases"
        elif str(cell_content).strip() and str(cell_content) != 'Vacío':
            return "Clases"
        else:
            return "No hay clases"
    for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
        df_horario[day] = df_horario[day].apply(get_activity_type)
    return df_horario


def discretize_temp_interna(df):
    bins = [15, 23, 30, 38]
    labels = [0, 1, 2]
    df['temp_discretizada'] = pd.cut(df['temp'], bins=bins, labels=labels, right=False)
    return df

def discretize_temp_externa(df):
    bins = [20, 24, 30, 35]
    labels = [0, 1, 2]
    df['temp_discretizada'] = pd.cut(df['temp'], bins=bins, labels=labels, right=False)
    return df

def discretize_estado_aire(df):
    df['estado_del_aire'] = (df['potencia_A'] > 20).astype(int)
    return df

def discretize_n_personas(df):
    def map_n_personas(val):
        if val == 0:
            return 0
        elif 1 <= val <= 10:
            return 1
        elif 11 <= val <= 20:
            return 2
        return None
    df['n_personas_discretizada'] = df['n_personas'].apply(map_n_personas)
    return df

def discretize_ubicacion(df):
    def map_ubicacion(val):
        if val == 'dispersas':
            return 0
        elif val == 'agrupadas cerca de ventilación':
            return 1
        elif val == 'agrupadas lejos':
            return 2
        return None
    df['ubicacion_discretizada'] = df['ubicacion'].apply(map_ubicacion)
    return df

def discretize_opinion_termica(df):
    def map_opinion(val):
        if val == "Frío":
            return 0
        elif val == "Cómodo":
            return 1
        elif val == "Calor":
            return 2
        return None
    df['opinion_termica_discretizada'] = df['opinion'].apply(map_opinion)
    return df


def build_df_features(dfs, flags):
    """
    Receives all dataframes and simulation flags from process_files.
    Simulates horario if needed, discretizes all non-empty dataframes, and merges them into one features table.
    Simulated variables are added at the end if needed.
    Returns the final df_features dataframe.
    """
    # Discretize all available dataframes
    result = {}
    # Horario
    if flags.get('horario', True):
        horario_df = simulate_horario_df()
    else:
        horario_df = dfs['horario']
    result['horario'] = horario_df

    # Helper to get first dataframe from list or None
    def get_df(key):
        val = dfs.get(key)
        if val is None:
            return None
        if isinstance(val, list):
            return val[0] if len(val) > 0 else None
        return val

    temp_interna_df = get_df('temp_interna') if not flags.get('temp_interna', True) else None
    temp_externa_df = get_df('temp_externa') if not flags.get('temp_externa', True) else None
    estado_aire_df = get_df('estado_aire') if not flags.get('estado_aire', True) else None
    personas_ubicacion_df = get_df('personas_ubicacion') if not flags.get('personas_ubicacion', True) else None
    opinion_termica_df = get_df('opinion_termica') if not flags.get('opinion_termica', True) else None

    if temp_interna_df is not None:
        temp_interna_df = discretize_temp_interna(temp_interna_df)
    if temp_externa_df is not None:
        temp_externa_df = discretize_temp_externa(temp_externa_df)
    if estado_aire_df is not None:
        estado_aire_df = discretize_estado_aire(estado_aire_df)
    if personas_ubicacion_df is not None:
        personas_ubicacion_df = discretize_n_personas(personas_ubicacion_df)
        personas_ubicacion_df = discretize_ubicacion(personas_ubicacion_df)
    if opinion_termica_df is not None:
        opinion_termica_df = discretize_opinion_termica(opinion_termica_df)

    # Progressive merge on ['Fecha', 'Hora']
    merge_keys = ['Fecha', 'Hora']
    dfs_to_merge = []
    if temp_interna_df is not None:
        dfs_to_merge.append(temp_interna_df)
    if temp_externa_df is not None:
        dfs_to_merge.append(temp_externa_df.reset_index(drop=True))
    if estado_aire_df is not None:
        dfs_to_merge.append(estado_aire_df)
    if personas_ubicacion_df is not None:
        dfs_to_merge.append(personas_ubicacion_df)
    if opinion_termica_df is not None:
        dfs_to_merge.append(opinion_termica_df)

    # Start merging
    if dfs_to_merge:
        merged_df = dfs_to_merge[0]
        for next_df in dfs_to_merge[1:]:
            merged_df = pd.merge(merged_df, next_df, on=merge_keys, how='outer')
    else:
        merged_df = pd.DataFrame()

    # Add horario info if possible (merge by Hora)
    if not horario_df.empty:
        merged_df = pd.merge(merged_df, horario_df, on='Hora', how='left')

    # Create 'sensedAt_final' from available columns
    sensedAt_cols = [col for col in merged_df.columns if col.startswith('sensedAt')]
    if sensedAt_cols:
        merged_df['sensedAt_final'] = merged_df[sensedAt_cols].bfill(axis=1).iloc[:, 0]
    else:
        merged_df['sensedAt_final'] = None

    # Rename and reorganize columns
    col_map = {
        'sensedAt_final': 'sensedAt',
        'temp_discretizada_x': 'temp_interna_discretizada',
        'temp_discretizada_y': 'temp_externa_discretizada',
        'estado_del_aire': 'estado_aire',
        'n_personas_discretizada': 'n_personas',
        'ubicacion_discretizada': 'ubicacion',
        'opinion_termica_discretizada': 'opinion_termica',
    }
    final_cols = ['sensedAt', 'Fecha', 'Hora', 'temp_interna_discretizada', 'temp_externa_discretizada', 'estado_aire', 'n_personas', 'ubicacion', 'opinion_termica']
    merged_df = merged_df.rename(columns=col_map)
    # Only keep columns that exist and ensure uniqueness
    final_cols = [col for col in final_cols if col in merged_df.columns]
    # Remove duplicate columns if any
    df_features = merged_df[final_cols].copy()
    df_features = df_features.loc[:,~df_features.columns.duplicated()]

    # Add simulated variables at the end if needed
    df_features = add_simulated_variables(df_features, flags)
    # Always add 'clases a continuación' using horario_df
    df_features = add_clases_a_continuacion(df_features, horario_df)
    return df_features

def add_clases_a_continuacion(df_features, df_horario, tiempo=30):
    """
    Adds the column 'clases a continuación' to df_features, indicating if there is a class in the next 'tiempo' minutes.
    df_horario must have columns: 'Hora', 'start_time', 'end_time', and day columns ('Lunes', ...).
    """
    # Limpieza inicial
    df_horario['Hora'] = df_horario['Hora'].str.strip()
    # Expandimos el rango de horas en una estructura más manejable
    def expand_range(row):
        start, end = row['Hora'].split('-')
        return pd.Series({
            'start_time': pd.to_datetime(start, format='%H:%M').time(),
            'end_time': pd.to_datetime(end, format='%H:%M').time()
        })
    df_horario[['start_time', 'end_time']] = df_horario.apply(expand_range, axis=1)
    # Añadir la columna "clases a continuación"
    df_features['clases a continuación'] = 0  # Falso
    # Convertir la columna 'Hora' de df_features a objetos time para comparación
    df_features['Hora_time'] = df_features['Hora'].apply(lambda x: pd.to_datetime(str(x)).time())
    # Convertir la columna 'Fecha' de df_features a datetime para obtener el día de la semana
    df_features['Fecha_datetime'] = pd.to_datetime(df_features['Fecha'])
    # Iterar sobre cada fila de df_features
    for index, row in df_features.iterrows():
        current_date = row['Fecha_datetime']
        current_time = row['Hora_time']
        day_of_week = current_date.strftime('%A')
        day_mapping = {
            'Monday': 'Lunes',
            'Tuesday': 'Martes',
            'Wednesday': 'Miércoles',
            'Thursday': 'Jueves',
            'Friday': 'Viernes',
            'Saturday': None,
            'Sunday': None
        }
        day_column = day_mapping.get(day_of_week)
        if day_column is not None:
            current_datetime = pd.to_datetime(f'{current_date.strftime("%Y-%m-%d")} {current_time.strftime("%H:%M:%S")}')
            future_datetime = current_datetime + pd.Timedelta(minutes=tiempo)
            future_time = future_datetime.time()
            current_minutes = current_time.hour * 60 + current_time.minute
            future_minutes = future_time.hour * 60 + future_time.minute
            for hor_index, hor_row in df_horario.iterrows():
                hor_start_time = hor_row['start_time']
                hor_end_time = hor_row['end_time']
                activity = hor_row[day_column]
                hor_start_minutes = hor_start_time.hour * 60 + hor_start_time.minute
                hor_end_minutes = hor_end_time.hour * 60 + hor_end_time.minute
                if future_minutes < current_minutes:
                    future_minutes += 24 * 60
                # Escenario 1: Una clase empieza dentro de los próximos 30 minutos.
                if current_minutes <= hor_start_minutes < future_minutes:
                    if activity == 'Clases':
                        df_features.at[index, 'clases a continuación'] = 1
                        break
                # Escenario 2: La hora actual está dentro de una clase que continúa durante los próximos 30 minutos.
                if hor_start_minutes <= current_minutes < hor_end_minutes:
                    if activity == 'Clases':
                        df_features.at[index, 'clases a continuación'] = 1
                        break
    # Eliminar las columnas temporales si ya no se necesitan
    df_features.drop(columns=['Hora_time', 'Fecha_datetime'], inplace=True)
    return df_features

def add_simulated_variables(df_features, flags):
    """
    Adds simulated variables to df_features if needed, using specified probabilities.
    Does NOT simulate horario.
    """
    np.random.seed(42)
    if flags.get('n_personas', True) and 'n_personas' not in df_features.columns:
        df_features['n_personas'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.3, 0.5, 0.2])
    if flags.get('ubicacion', True) and 'ubicacion' not in df_features.columns:
        df_features['ubicacion'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.5, 0.25, 0.25])
    if flags.get('opinion_termica', True) and 'opinion_termica' not in df_features.columns:
        df_features['opinion_termica'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.2, 0.6, 0.2])
    if flags.get('temp_interna_discretizada', True) and 'temp_interna_discretizada' not in df_features.columns:
        df_features['temp_interna_discretizada'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.2, 0.6, 0.2])
    if flags.get('temp_externa_discretizada', True) and 'temp_externa_discretizada' not in df_features.columns:
        df_features['temp_externa_discretizada'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.2, 0.6, 0.2])
    if flags.get('estado_aire', True) and 'estado_aire' not in df_features.columns:
        df_features['estado_aire'] = np.random.choice([0, 1], size=len(df_features), p=[0.2, 0.8])
    return df_features
