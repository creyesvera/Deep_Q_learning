# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np



class data_preprocessing():
    def __init__(self, fechas = None, file_path_horario='data/Horario de clases.xlsx'):
        self.df_features = None
        if fechas is not None:
            df_estacion, df_shelly_iz, df_temp = self.load_data(fechas)
        else:
            df_estacion = pd.read_csv('data/df_estacion.csv')
            df_shelly_iz = pd.read_csv('data/df_shelly_iz.csv')
            df_temp = pd.read_csv('data/df_temp.csv')

        self.unify_Hayiot_data(self, df_estacion, df_shelly_iz, df_temp)

        #self.df_features deja de ser None

        df_horario = self.load_schedule(self, file_path_horario)
        self.simulate_n_personas(self, df_horario)
        self.simulate_person_locations(self)
        self.simulate_thermal_opinion(self)

        #output_file_path = 'LIVE_data/df_features.csv'
        #self.df_features.to_csv(output_file_path, index=False)

        return self.df_features

    def load_data(fechas):

        df_estacion = pd.read_csv('data/df_estacion.csv')
        df_shelly_iz = pd.read_csv('data/df_shelly_iz.csv')
        df_temp = pd.read_csv('data/df_temp.csv')

        return df_estacion, df_shelly_iz, df_temp

    # =============================================================================================
    # =============================================================================================
    # Archivo de HAYIOT cargados en 
    # file_path_estacion = '/content/Deep_Q_learning/data/df_estacion.csv'
    # file_path_shelly_iz = '/content/Deep_Q_learning/data/df_shelly_iz.csv'
    # file_path_temp = '/content/Deep_Q_learning/data/df_temp.csv'
    # =============================================================================================
    # =============================================================================================


    def unify_Hayiot_data(self, df_estacion, df_shelly_iz, df_temp):
        #df_estacion = pd.read_csv(file_path_estacion)
        #df_shelly_iz = pd.read_csv(file_path_shelly_iz)
        #df_temp = pd.read_csv(file_path_temp)

        list_sensedAt= pd.concat([df_estacion['sensedAt'],df_shelly_iz['sensedAt'],df_temp['sensedAt']]).unique()
        np.sort(list_sensedAt)


        for df in [df_estacion, df_shelly_iz, df_temp]:
            df['sensedAt'] = pd.to_datetime(df['sensedAt'], format='mixed')

        all_sensedAt = (
            pd.concat([
                df_estacion['sensedAt'],
                df_shelly_iz['sensedAt'],
                df_temp['sensedAt']
            ])
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        self.df_features = (
            pd.DataFrame({'sensedAt': sorted(all_sensedAt)})
            .reset_index(drop=True)
        )
        self.df_features = self.df_features.merge(
            df_estacion[['sensedAt', 'temp']]
                .rename(columns={'temp': 'temp_externa'}),
            on='sensedAt',
            how='left'
        )
        self.df_features = self.df_features.merge(
            df_temp[['sensedAt', 'temp']]
                .rename(columns={'temp': 'temp_interna'}),
            on='sensedAt',
            how='left'
        )
        self.df_features = self.df_features.merge(
            df_shelly_iz[['sensedAt', 'potencia_A']],
            on='sensedAt',
            how='left'
        )


        self.df_features['Fecha'] = self.df_features['sensedAt'].dt.date
        self.df_features['Hora']  = self.df_features['sensedAt'].dt.time
        self.df_features = self.df_features[
            ['sensedAt', 'Fecha', 'Hora',
            'temp_externa', 'temp_interna', 'potencia_A']
        ]

        self.df_features['estado del aire'] = (self.df_features['potencia_A'] > 20).astype(int)

        return None

    # =============================================================================================
    # =============================================================================================
    # Archivo de horario cargado en 
    # file_path_horario = '/content/Deep_Q_learning/data/Horario de clases.xlsx'
    # =============================================================================================
    # =============================================================================================

    def time_to_minutes(t):
        h, m = map(int, t.split(':'))
        return h * 60 + m

    # ==============================
    # MAPEO DE ACTIVIDADES
    # ==============================
    def map_activity(val):
        if pd.isna(val):
            return 0
        if val == 'Laboratorio abierto':
            return 1
        if val == 'Clases practicas':
            return 2
        return 0




    def load_schedule(self, file_path_horario='data/Horario de clases.xlsx'):

        df_horario = pd.read_excel(file_path_horario)


        days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']

        for day in days:
            # Replace values that are not null and not 'Laboratorio abierto' with 'Clase'
            mask = (df_horario[day].notna()) & (df_horario[day] != 'Laboratorio abierto')
            df_horario.loc[mask, day] = 'Clases practicas'



        df_horario[['Inicio', 'Fin']] = df_horario['Hora'].str.split('-', expand=True)
        df_horario = df_horario.drop(columns=['Hora'])

        print("df_horario with 'Inicio' and 'Fin':")

        # ==============================
        # PREPARAR df_horario
        # ==============================

        tiempo = 30



        df_horario['inicio_min'] = df_horario['Inicio'].apply(self.time_to_minutes)
        df_horario['fin_min'] = df_horario['Fin'].apply(self.time_to_minutes)



        # ==============================
        # PREPARAR self.df_features
        # ==============================


        self.df_features['datetime'] = pd.to_datetime(self.df_features['sensedAt'])
        self.df_features['minutes'] = (
            self.df_features['datetime'].dt.hour * 60 +
            self.df_features['datetime'].dt.minute
        )
        self.df_features['future_minutes'] = self.df_features['minutes'] + tiempo

        day_map = {
            0: 'Lunes',
            1: 'Martes',
            2: 'Miércoles',
            3: 'Jueves',
            4: 'Viernes'
        }

        self.df_features['day_col'] = self.df_features['datetime'].dt.weekday.map(day_map)


        
        # ==============================
        # CÁLCULO PRINCIPAL
        # ==============================
        self.df_features['clases_a_continuacion'] = 0

        for _, hrow in df_horario.iterrows():
            start = hrow['inicio_min']
            end = hrow['fin_min']

            overlap = (                                               ###Detectar solapamiento temporal
                (self.df_features['minutes'] < end) &                      ###Esto detecta si la ventana futura intersecta una clase.
                (self.df_features['future_minutes'] > start)               ###Ejemplo: ahora = 09:40, futuro = 10:10, clase = 09:00–10:00 -> Sí hay solapamiento
            )

            for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
                mask = overlap & (self.df_features['day_col'] == day)
                clase = self.map_activity(hrow[day])

                self.df_features.loc[mask, 'clases_a_continuacion'] = (
                    self.df_features.loc[mask, 'clases_a_continuacion']
                    .clip(lower=clase)
                )

        # ==============================
        # LIMPIEZA
        # ==============================
        self.df_features.drop(
            columns=['datetime', 'minutes', 'future_minutes', 'day_col'],
            inplace=True
        )

        return df_horario


    # =============================================================================================
    # =============================================================================================
    # Simulación de número de personas en el aula
    # =============================================================================================
    # =============================================================================================

    def actividad_actual(row, df_horario):
        dia = row['day_col']

        # Fin de semana o día no mapeado
        if pd.isna(dia):
            return np.nan

        minuto = row['minutes']

        bloque = df_horario[
            (df_horario['inicio_min'] <= minuto) &
            (df_horario['fin_min'] > minuto)
        ]

        if bloque.empty:
            return np.nan

        # Puede haber NaN en el horario (y eso es válido)
        return bloque.iloc[0][dia]

    def personas_objetivo(actividad):
        if actividad == 'Clases practicas':
            return np.random.randint(8, 16)

        if actividad == 'Laboratorio abierto':
            return np.random.randint(0, 6)

        # NaN / sin actividad
        return np.random.choice(
            [0, 1, 2, 3, 4],
            p=[0.80, 0.10, 0.06, 0.03, 0.01]
        )

    def transicion_suave(actual, objetivo, max_delta=3):
        if actual < objetivo:
            return min(actual + np.random.randint(1, max_delta + 1), objetivo)
        if actual > objetivo:
            return max(actual - np.random.randint(1, max_delta + 1), objetivo)
        return actual


    def simulate_n_personas(self, df_horario):
        # ==============================
        # RECREAR VARIABLES NECESARIAS
        # ==============================


        self.df_features['datetime'] = pd.to_datetime(self.df_features['sensedAt'])
        self.df_features['minutes'] = (
            self.df_features['datetime'].dt.hour * 60 +
            self.df_features['datetime'].dt.minute
        )

        self.df_features['day_col'] = self.df_features['datetime'].dt.weekday.map({
            0: 'Lunes',
            1: 'Martes',
            2: 'Miércoles',
            3: 'Jueves',
            4: 'Viernes'
        })


        n_personas = np.zeros(len(self.df_features), dtype=int)
        n_personas[0] = 0

        for i in range(1, len(self.df_features)):
            actividad = self.actividad_actual(self.df_features.loc[i], df_horario)
            objetivo = self.personas_objetivo(actividad)

            n_personas[i] = self.transicion_suave(
                actual=n_personas[i - 1],
                objetivo=objetivo,
                max_delta=3
            )

        self.df_features['n_personas'] = n_personas

        self.df_features = self.df_features.drop(columns=['datetime', 'minutes', 'day_col'])


    # =============================================================================================
    # =============================================================================================
    # Simulación de ubicación de personas en el aula
    # Archivo en 
    # directory_path = '/content/Deep_Q_learning/data/analisis en R_data_personas/Personas'
    # =============================================================================================
    # =============================================================================================

    def simulate_person_locations(self, directory_path = 'data\analisis en R_data_personas\Personas'):

        all_files_and_dirs = os.listdir(directory_path)

        csv_files = [f for f in all_files_and_dirs if f.endswith('.csv')]

        csv_files_sorted = sorted(csv_files)

        # Initialize an empty dictionary to store DataFrames
        dataframes_dict = {}

        print(f"CSV files in '{directory_path}' (orden ascendente):")
        for csv_file in csv_files_sorted:
            print(csv_file)

            # Construct the full path to the CSV file
            full_file_path = os.path.join(directory_path, csv_file)

            # Generate a dynamic DataFrame name (e.g., df_523008insert)
            df_name = 'df_' + os.path.splitext(csv_file)[0]

            # Load the CSV file into a DataFrame and store it in the dictionary
            dataframes_dict[df_name] = pd.read_csv(full_file_path)

            print(f"Loaded {csv_file} into DataFrame: {df_name}")

        print("\nAll CSV files loaded into DataFrames in 'dataframes_dict'.")

        # =====================================================
        # ASIGNACIÓN SECUENCIAL DE RUTAS A self.df_features
        # USANDO dataframes_dict (YA CARGADO)
        # =====================================================

        import random
        import pandas as pd

        # -----------------------------------------------------
        # 1. PREPARAR RUTAS (una por df_5230**insert)
        # -----------------------------------------------------
        # dataframes_dict ya existe y contiene:
        # { 'df_523001insert': df, 'df_523002insert': df, ... }

        rutas = {}

        for key, df_ruta in dataframes_dict.items():
            df_ruta = df_ruta.copy()
            df_ruta['Time'] = pd.to_datetime(df_ruta['Time'])
            rutas[key] = df_ruta

        # -----------------------------------------------------
        # 2. PREPARAR self.df_features
        # -----------------------------------------------------
        df = self.df_features.copy()
        df['sensedAt'] = pd.to_datetime(df['sensedAt'])

        # columna ubicacion: lista de duplas (x, y)
        df['ubicacion'] = [[] for _ in range(len(df))]

        # -----------------------------------------------------
        # 3. POOL DE RUTAS SIN REPETICIÓN
        # -----------------------------------------------------
        rutas_ids = list(rutas.keys())
        random.shuffle(rutas_ids)

        rutas_disponibles = rutas_ids.copy()
        rutas_usadas = []

        def tomar_ruta():
            global rutas_disponibles, rutas_usadas
            if len(rutas_disponibles) == 0:
                rutas_disponibles = rutas_usadas.copy()
                rutas_usadas = []
                random.shuffle(rutas_disponibles)
            r = rutas_disponibles.pop(0)
            rutas_usadas.append(r)
            return r

        # -----------------------------------------------------
        # 4. ALGORITMO PRINCIPAL (TU PSEUDOCÓDIGO)
        # -----------------------------------------------------
        max_personas = int(df['n_personas'].max())

        for k in range(1, max_personas + 1):

            i = 0
            while i < len(df):

                # buscar inicio de una nueva persona virtual
                if df.loc[i, 'n_personas'] >= k:

                    ruta_id = tomar_ruta()
                    ruta = rutas[ruta_id]
                    j = 0  # índice dentro de la ruta

                    # avanzar mientras la persona "exista"
                    while (
                        i < len(df) and
                        j < len(ruta) and
                        df.loc[i, 'n_personas'] >= k
                    ):
                        x = ruta.iloc[j]['world_x']
                        y = ruta.iloc[j]['world_y']

                        df.at[i, 'ubicacion'].append((x, y))

                        i += 1
                        j += 1
                else:
                    i += 1

        # -----------------------------------------------------
        # 5. VERIFICACIÓN DE CONSISTENCIA
        # -----------------------------------------------------
        df['check'] = df['ubicacion'].apply(len) == df['n_personas']
        print(df['check'].value_counts())

        # df es el self.df_features final con ubicaciones asignadas

        self.df_features['ubicacion'] = df['ubicacion']


        # =============================================================================================
        # =============================================================================================
        # UBICACIÓN RELATIVA DENTRO DEL AULA
        # =============================================================================================
        # =============================================================================================


        # ==============================
        # PARÁMETROS
        # ==============================
        DISP_TH = 3.0  # umbral de dispersión (ajústalo si es necesario)

        # Rectángulos (x_min, x_max, y_min, y_max)
        RECT_1 = (0.0, 18.0, -7.5, 1.0)
        RECT_2 = (0.0, 18.0, -17.5, -7.5)

        # ==============================
        # FUNCIÓN PRINCIPAL
        # ==============================
        def clasificar_ubicacion(lista_puntos):
            if not lista_puntos or len(lista_puntos) == 0:
                return 0

            xs = np.array([p[0] for p in lista_puntos], dtype=float)
            ys = np.array([p[1] for p in lista_puntos], dtype=float)

            mean_x, mean_y = xs.mean(), ys.mean()
            std_x, std_y = xs.std(), ys.std()

            # Dispersión alta
            if std_x >= DISP_TH or std_y >= DISP_TH:
                return 0

            # Dispersión baja → verificar rectángulos
            if (RECT_1[0] <= mean_x <= RECT_1[1]) and (RECT_1[2] <= mean_y <= RECT_1[3]):
                return 1

            if (RECT_2[0] <= mean_x <= RECT_2[1]) and (RECT_2[2] <= mean_y <= RECT_2[3]):
                return 2

            return 0

        # ==============================
        # APLICAR AL DATAFRAME
        # ==============================
        self.df_features['ubicacion_relativa'] = self.df_features['ubicacion'].apply(clasificar_ubicacion)

        import numpy as np

        self.df_features['ubicacion'] = self.df_features['ubicacion'].apply(lambda x: np.nan if len(x) == 0 else x)
        self.df_features.loc[self.df_features['ubicacion'].isna(), 'ubicacion_relativa'] = np.nan

        """## fin de simulación de ubicación de personas

        #Añadir datos de opinión

        | Variable                         |  Rango discretizado (niveles)                         | N niveles |
        | :------------------------------- |:--------------------------------------------------- | :-------- |
        | Panel de opinión térmica       | 0: muy fría, 1: neutra, 2: muy calurosa | 3         |
        """

        return None

    # =============================================================================================
    # =============================================================================================
    # Simulación de opinión térmica
    # =============================================================================================
    # =============================================================================================

    def simulate_thermal_opinion(self):

        # Inicializar la columna como NaN
        self.df_features['thermal_opinion'] = np.nan

        # Máscara: solo filas con personas
        mask_personas = self.df_features['n_personas'] > 0

        # Asignar valores aleatorios solo a esas filas
        self.df_features.loc[mask_personas, 'thermal_opinion'] = np.random.choice(
            [0, 1, 2],
            size=mask_personas.sum(),
            p=[0.2, 0.6, 0.2]
        )

        return None


    # =============================================================================================
    # =============================================================================================
    # FIN DE LA CLASE data_preprocessing
    # =============================================================================================
    # =============================================================================================