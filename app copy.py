import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from src.data_preprocessing import build_df_features

def validate_horario(file):
    # Custom validation for Horario
    try:
        df = pd.read_excel(file)
        # Add your validation logic here
        return df, "Horario válido"
    except Exception as e:
        return None, f"Error en Horario: {e}"

def archivo_a_dataframe(file_like):
    df = pd.read_csv(file_like)
    df['sensedAt'] = pd.to_datetime(df['sensedAt'])
    df_pivot = df.pivot(index='sensedAt', columns='type', values='data')
    df_pivot.reset_index(inplace=True)
    df_pivot['Fecha'] = df_pivot['sensedAt'].dt.date
    df_pivot['Hora'] = df_pivot['sensedAt'].dt.time
    return df_pivot

def preprocessing(df, columnas_a_filtrar, limites_outliers):
    df['sensedAt'] = pd.to_datetime(df['sensedAt'])
    df.sort_values('sensedAt', inplace=True)
    df.set_index('sensedAt', inplace=True)
    df = df.infer_objects(copy=False)
    df_interpolated = df.interpolate(method='time')
    df_interpolated = df_interpolated.bfill()
    df_interpolated = df_interpolated.ffill()
    for col, (min_val, max_val) in limites_outliers.items():
        if col in df_interpolated.columns:
            df_interpolated = df_interpolated[df_interpolated[col].between(min_val, max_val)]
    return df_interpolated

# Update validate_temp_interna to process all valid files

def validate_temp_interna(file):
    columnas = ['temp']
    limites = {'temp': (15, 38)}
    columnas_a_conservar = ['Fecha', 'Hora', 'sensedAt', 'temp']
    processed_dfs = []
    try:
        if zipfile.is_zipfile(file):
            temp_results = []
            for_df_scaled = []
            with zipfile.ZipFile(file) as z:
                for fname in z.namelist():
                    with z.open(fname) as f:
                        # Read CSV directly from file pointer
                        f.seek(0)
                        df = pd.read_csv(f)
                        if not all(col in df.columns for col in ['type', 'sensedAt']):
                            temp_results.append(f"{fname}: Faltan columnas 'type' o 'sensedAt'")
                            continue
                        if 'temp' not in df['type'].values:
                            temp_results.append(f"{fname}: No hay filas con type='temp'")
                            continue
                        temp_results.append(f"{fname}: OK")
                        # Filter only rows with type 'temp'
                        df_temp = df[df['type'] == 'temp'].copy()
                        df_temp['sensedAt'] = pd.to_datetime(df_temp['sensedAt'])
                        df_temp = df_temp[['sensedAt', 'type', 'data']]
                        df_temp = df_temp.rename(columns={'data': 'temp'})
                        # Pivot not needed, just keep temp
                        df_temp['Fecha'] = df_temp['sensedAt'].dt.date
                        df_temp['Hora'] = df_temp['sensedAt'].dt.time
                        # Keep only required columns
                        for col in columnas_a_conservar:
                            if col not in df_temp.columns:
                                df_temp[col] = None
                        df_temp = df_temp[columnas_a_conservar]
                        print(df_temp.head())
                        df_scaled = preprocessing(df_temp, columnas_a_filtrar=columnas, limites_outliers=limites)
                        for_df_scaled.append(df_scaled.reset_index())
            status = '\n'.join(temp_results)
            # Merge all dataframes and group if there are many
            if for_df_scaled:
                df_combinado = pd.concat(for_df_scaled)
                df_temperatura_interna = df_combinado.groupby(['sensedAt', 'Fecha', 'Hora']).agg(temp=('temp', 'mean')).reset_index()
                return [df_temperatura_interna], status
            else:
                return None, status
        else:
            file.seek(0)
            df = pd.read_csv(file)
            if not all(col in df.columns for col in ['type', 'sensedAt']):
                return None, "Faltan columnas 'type' o 'sensedAt'"
            if 'temp' not in df['type'].values:
                return None, "No hay filas con type='temp'"
            df_temp = df[df['type'] == 'temp'].copy()
            df_temp['sensedAt'] = pd.to_datetime(df_temp['sensedAt'])
            df_temp = df_temp[['sensedAt', 'type', 'data']]
            df_temp = df_temp.rename(columns={'data': 'temp'})
            df_temp['Fecha'] = df_temp['sensedAt'].dt.date
            df_temp['Hora'] = df_temp['sensedAt'].dt.time
            for col in columnas_a_conservar:
                if col not in df_temp.columns:
                    df_temp[col] = None
            df_temp = df_temp[columnas_a_conservar]
            print(df_temp.head())
            df_scaled = preprocessing(df_temp, columnas_a_filtrar=columnas, limites_outliers=limites)
            # If only one, just return the processed dataframe
            return [df_scaled.reset_index()], "Temperatura Interna válida"
    except Exception as e:
        return None, f"Error en Temperatura Interna: {e}"

def validate_temp_externa(file):
    columnas = ['temp']
    limites = {'temp': (15, 40)}
    columnas_a_conservar = ['Fecha', 'Hora', 'sensedAt', 'temp']
    processed_dfs = []
    try:
        if zipfile.is_zipfile(file):
            temp_results = []
            for_df_scaled = []
            with zipfile.ZipFile(file) as z:
                for fname in z.namelist():
                    with z.open(fname) as f:
                        # Read CSV directly from file pointer
                        f.seek(0)
                        df = pd.read_csv(f)
                        if not all(col in df.columns for col in ['type', 'sensedAt']):
                            temp_results.append(f"{fname}: Faltan columnas 'type' o 'sensedAt'")
                            continue
                        if 'temp' not in df['type'].values:
                            temp_results.append(f"{fname}: No hay filas con type='temp'")
                            continue
                        temp_results.append(f"{fname}: OK")
                        # Filter only rows with type 'temp'
                        df_temp = df[df['type'] == 'temp'].copy()
                        df_temp['sensedAt'] = pd.to_datetime(df_temp['sensedAt'])
                        df_temp = df_temp[['sensedAt', 'type', 'data']]
                        df_temp = df_temp.rename(columns={'data': 'temp'})
                        df_temp['Fecha'] = df_temp['sensedAt'].dt.date
                        df_temp['Hora'] = df_temp['sensedAt'].dt.time
                        for col in columnas_a_conservar:
                            if col not in df_temp.columns:
                                df_temp[col] = None
                        df_temp = df_temp[columnas_a_conservar]
                        print(df_temp.head())
                        df_scaled = preprocessing(df_temp, columnas_a_filtrar=columnas, limites_outliers=limites)
                        for_df_scaled.append(df_scaled.reset_index())
            status = '\n'.join(temp_results)
            if for_df_scaled:
                df_combinado = pd.concat(for_df_scaled)
                df_temperatura_externa = df_combinado.groupby(['sensedAt', 'Fecha', 'Hora']).agg(temp=('temp', 'mean')).reset_index()
                return [df_temperatura_externa], status
            else:
                return None, status
        else:
            file.seek(0)
            df = pd.read_csv(file)
            if not all(col in df.columns for col in ['type', 'sensedAt']):
                return None, "Faltan columnas 'type' o 'sensedAt'"
            if 'temp' not in df['type'].values:
                return None, "No hay filas con type='temp'"
            df_temp = df[df['type'] == 'temp'].copy()
            df_temp['sensedAt'] = pd.to_datetime(df_temp['sensedAt'])
            df_temp = df_temp[['sensedAt', 'type', 'data']]
            df_temp = df_temp.rename(columns={'data': 'temp'})
            df_temp['Fecha'] = df_temp['sensedAt'].dt.date
            df_temp['Hora'] = df_temp['sensedAt'].dt.time
            for col in columnas_a_conservar:
                if col not in df_temp.columns:
                    df_temp[col] = None
            df_temp = df_temp[columnas_a_conservar]
            print(df_temp.head())
            df_scaled = preprocessing(df_temp, columnas_a_filtrar=columnas, limites_outliers=limites)
            return [df_scaled.reset_index()], "Temperatura Externa válida"
    except Exception as e:
        return None, f"Error en Temperatura Externa: {e}"

def validate_opinion_termica(file):
    # Custom validation for Opinión térmica
    try:
        file.seek(0)
        df = pd.read_csv(file)
        # Add your validation logic here
        return df, "Opinión térmica válida"
    except Exception as e:
        return None, f"Error en Opinión térmica: {e}"

def validate_personas_ubicacion(file):
    # Custom validation for Número de personas y Ubicación
    try:
        file.seek(0)
        df = pd.read_csv(file)
        # Add your validation logic here
        return df, "Número de personas y Ubicación válido"
    except Exception as e:
        return None, f"Error en Número de personas y Ubicación: {e}"

def validate_estado_aire(file):
    columnas_a_conservar = ['sensedAt','Fecha', 'Hora', 'potencia_A']
    processed_dfs = []
    try:
        if zipfile.is_zipfile(file):
            results = []
            for_df = []
            with zipfile.ZipFile(file) as z:
                for fname in z.namelist():
                    with z.open(fname) as f:
                        f.seek(0)
                        df = pd.read_csv(f)
                        if not all(col in df.columns for col in ['type', 'sensedAt']):
                            results.append(f"{fname}: Faltan columnas 'type' o 'sensedAt'")
                            continue
                        if 'potencia_A' not in df['type'].values:
                            results.append(f"{fname}: No hay filas con type='potencia_A'")
                            continue
                        results.append(f"{fname}: OK")
                        # Filter only rows with type 'potencia_A'
                        df_pot = df[df['type'] == 'potencia_A'].copy()
                        df_pot['sensedAt'] = pd.to_datetime(df_pot['sensedAt'])
                        df_pot = df_pot[['sensedAt', 'type', 'data']]
                        df_pot = df_pot.rename(columns={'data': 'potencia_A'})
                        df_pot['Fecha'] = df_pot['sensedAt'].dt.date
                        df_pot['Hora'] = df_pot['sensedAt'].dt.time
                        for col in columnas_a_conservar:
                            if col not in df_pot.columns:
                                df_pot[col] = None
                        df_pot = df_pot[columnas_a_conservar]
                        print(df_pot.head())
                        for_df.append(df_pot)
            status = '\n'.join(results)
            if for_df:
                df_combinado = pd.concat(for_df)
                df_shelly_iz = df_combinado.reset_index(drop=True)
                return [df_shelly_iz], status
            else:
                return None, status
        else:
            file.seek(0)
            df = pd.read_csv(file)
            if not all(col in df.columns for col in ['type', 'sensedAt']):
                return None, "Faltan columnas 'type' o 'sensedAt'"
            if 'potencia_A' not in df['type'].values:
                return None, "No hay filas con type='potencia_A'"
            df_pot = df[df['type'] == 'potencia_A'].copy()
            df_pot['sensedAt'] = pd.to_datetime(df_pot['sensedAt'])
            df_pot = df_pot[['sensedAt', 'type', 'data']]
            df_pot = df_pot.rename(columns={'data': 'potencia_A'})
            df_pot['Fecha'] = df_pot['sensedAt'].dt.date
            df_pot['Hora'] = df_pot['sensedAt'].dt.time
            for col in columnas_a_conservar:
                if col not in df_pot.columns:
                    df_pot[col] = None
            df_pot = df_pot[columnas_a_conservar]
            print(df_pot.head())
            return [df_pot.reset_index(drop=True)], "Estado del Aire acondicionado válido"
    except Exception as e:
        return None, f"Error en Estado del Aire acondicionado: {e}"

def process_files(horario, temp_interna, temp_externa, opinion_termica, personas_ubicacion, estado_aire):
    results = {}
    dfs = {}
    simulate_flags = {}
    # Validate each file
    dfs['horario'], results['horario'] = validate_horario(horario)
    simulate_flags['horario'] = dfs['horario'] is None
    dfs['temp_interna'], results['temp_interna'] = validate_temp_interna(temp_interna)
    simulate_flags['temp_interna'] = dfs['temp_interna'] is None
    dfs['temp_externa'], results['temp_externa'] = validate_temp_externa(temp_externa)
    simulate_flags['temp_externa'] = dfs['temp_externa'] is None
    dfs['opinion_termica'], results['opinion_termica'] = validate_opinion_termica(opinion_termica)
    simulate_flags['opinion_termica'] = dfs['opinion_termica'] is None
    dfs['personas_ubicacion'], results['personas_ubicacion'] = validate_personas_ubicacion(personas_ubicacion)
    simulate_flags['personas_ubicacion'] = dfs['personas_ubicacion'] is None
    dfs['estado_aire'], results['estado_aire'] = validate_estado_aire(estado_aire)
    simulate_flags['estado_aire'] = dfs['estado_aire'] is None
    # Show status for each file
    status = '\n'.join([f"{k}: {v} | Simular: {simulate_flags[k]}" for k, v in results.items()])
    # Show preview of each dataframe (handle lists)
    def get_preview(df):
        if df is None:
            return None
        if isinstance(df, list):
            if len(df) > 0:
                return df[0].head(10)
            else:
                return None
        return df.head(10)
    previews = {k: get_preview(df) for k, df in dfs.items()}
    # Build df_features using builder
    df_features = build_df_features(dfs, simulate_flags)
    # Drop 'sensedAt' column if present
    if 'sensedAt' in df_features.columns:
        df_features = df_features.drop(columns=['sensedAt'])
    # Impute missing values with mode for each column
    for col in df_features.columns:
        if df_features[col].isna().any():
            moda = df_features[col].mode()[0]
            df_features[col] = df_features[col].fillna(moda)
    print("\n\nDF_FEATURES FULL COLUMNS:", df_features.columns.tolist())
    print(df_features.head(30))
    return status, previews['horario'], previews['temp_interna'], previews['temp_externa'], previews['opinion_termica'], previews['personas_ubicacion'], previews['estado_aire'], df_features.head(20)


st.title("Carga y validación de archivos para el sistema térmico")

archivo_horario = st.file_uploader("Horario (xlsx)", type=["xlsx"])
archivo_temp_interna = st.file_uploader("Temperatura Interna (csv/zip)", type=["csv", "zip"])
archivo_temp_externa = st.file_uploader("Temperatura Externa (csv/zip)", type=["csv", "zip"])
archivo_opinion = st.file_uploader("Opinión térmica (csv/zip)", type=["csv", "zip"])
archivo_personas = st.file_uploader("Número de personas y Ubicación (csv/zip)", type=["csv", "zip"])
archivo_estado_aire = st.file_uploader("Estado del Aire acondicionado (csv/zip)", type=["csv", "zip"])

if st.button("Procesar archivos"):
    status, preview_horario, preview_temp_interna, preview_temp_externa, preview_opinion, preview_personas, preview_estado, df_features_preview = process_files(
        archivo_horario, archivo_temp_interna, archivo_temp_externa, archivo_opinion, archivo_personas, archivo_estado_aire
    )

    st.subheader("Estado de validación")
    st.text(status)

    st.subheader("Previsualización de datos")
    st.write("Horario")
    st.dataframe(preview_horario)
    st.write("Temperatura Interna")
    st.dataframe(preview_temp_interna)
    st.write("Temperatura Externa")
    st.dataframe(preview_temp_externa)
    st.write("Opinión térmica")
    st.dataframe(preview_opinion)
    st.write("Número de personas y Ubicación")
    st.dataframe(preview_personas)
    st.write("Estado del Aire acondicionado")
    st.dataframe(preview_estado)
    st.write("df_features")
    st.dataframe(df_features_preview)

    # Realizar predicción e interpretación inmediatamente después de procesar archivos
    from prediction import using_trained_model,train_q_learning
    try:
        pred_result = train_q_learning(df_features_preview)
        st.subheader("Resultado de la predicción")
        st.dataframe(pred_result)

        # If ACTIONS is not available, use a default
        try:
            from prediction import ACTIONS
        except ImportError:
            ACTIONS = {0: "Mantener apagado", 1: "Encender"}

        st.subheader("🔍 Interpretación detallada de los primeros 20 estados")
        for idx, row in pred_result.head(20).iterrows():
            description = []
            # Usar los nombres de columna reales del dataframe
            columnas_estado = [
                'temp_interna_discretizada',
                'n_personas',
                'ubicacion',
                'opinion_termica',
                'clases a continuación',
                'temp_externa_discretizada',
                'estado_aire'
            ]
            variable_names = [
                "Temp. int", "N° personas", "Ubicación",
                "Opinión térmica", "Horario dentro de 30min", "Temp. ext", "AC"
            ]
            state_labels = {
                0: {0: "baja (15–23 °C)", 1: "media (24–30 °C)", 2: "alta (31–38 °C)"},  # temp_int
                1: {0: "0 personas", 1: "1–5 personas", 2: "6–10 personas"},              # n_people
                2: {0: "dispersas", 1: "agrupadas cerca ventilación", 2: "agrupadas lejos"},  # location
                3: {0: "muy fría (0–1)", 1: "neutra (2–3)", 2: "muy calurosa (4–5)"},     # opinion
                4: {0: "no hay clase", 1: "hay clase"},                                 # schedule
                5: {0: "baja (20–24 °C)", 1: "media (25–30 °C)", 2: "alta (31–35 °C)"},  # temp_ext
                6: {0: "apagado", 1: "encendido"}                                       # ac_status
            }
            for i, col in enumerate(columnas_estado):
                val = row.get(col, None)
                if pd.isna(val):
                    label = "NaN"
                else:
                    label = state_labels[i].get(val, str(val))
                description.append(f"{variable_names[i]}: {label}")
            # Acción recomendada
            action_label = row.get('accion_recomendada', "Sin acción")
            st.write(" | ".join(description))
            st.write(f"→ Acción recomendada: {action_label}")
            st.markdown("---")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
