import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# --- Configuración Inicial DEAP ---
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Funciones de Carga y Procesamiento de Datos ---
@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is None: return None, {}, {}, {}, [], {}, 0, {}
    try:
        df = pd.read_csv(uploaded_file)
        if 'Numero' not in df.columns or 'Atraso' not in df.columns or 'Frecuencia' not in df.columns:
            st.error("El archivo debe contener las columnas 'Numero', 'Atraso' y 'Frecuencia'."); return None, {}, {}, {}, [], {}, 0, {}
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce'); df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce'); df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
        df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
        df['Numero'], df['Atraso'], df['Frecuencia'] = df['Numero'].astype(int).astype(str), df['Atraso'].astype(int), df['Frecuencia'].astype(int)
        st.success("Archivo de datos cargado exitosamente.")
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso'])); numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atrasos_disponibles_int = sorted(df['Atraso'].unique()); numeros_validos = list(numero_a_atraso.keys())
        distribucion_probabilidad = {num: 1.0/len(numeros_validos) for num in numeros_validos} if numeros_validos else {}
        atraso_counts = df['Atraso'].astype(str).value_counts().to_dict(); total_atraso_dataset = df['Atraso'].sum()
        atraso_stats = {"min": df['Atraso'].min(), "max": df['Atraso'].max(), "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)}
        return df, numero_a_atraso, numero_a_frecuencia, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos: {e}"); return None, {}, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is None: return []
    try:
        df_hist = pd.read_csv(uploaded_file, header=None)
        historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 6]
        if historical_sets: st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones.")
        else: st.warning("El archivo de historial no contenía combinaciones válidas.")
        return historical_sets
    except Exception as e:
        st.error(f"Error al procesar el archivo de historial: {e}"); return []

# --- Funciones de Análisis Histórico ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    if not historical_sets or total_atraso_dataset is None: return None
    values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": int(np.mean(values)), "std": int(np.std(values))}

@st.cache_data
def analyze_historical_frequency_cv(historical_sets, numero_a_frecuencia):
    if not historical_sets or not numero_a_frecuencia: return None
    cv_values = [np.std(freqs) / np.mean(freqs) for s in historical_sets if (freqs := [numero_a_frecuencia.get(str(num), 0) for num in s]) and np.mean(freqs) > 0]
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

@st.cache_data
def analyze_historical_structure(historical_sets):
    if not historical_sets: return None, None, None
    sums = [sum(s) for s in historical_sets]; parity_counts = Counter(sum(1 for num in s if num % 2 == 0) for s in historical_sets)
    consecutive_counts = []
    for s in historical_sets:
        nums = sorted(list(s)); max_consecutive = 0; current_consecutive = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        consecutive_counts.append(max(max_consecutive, current_consecutive))
    sum_stats = {"min": int(np.min(sums)), "max": int(np.max(sums)), "mean": int(np.mean(sums)), "std": int(np.std(sums))}
    return sum_stats, parity_counts, Counter(consecutive_counts)

@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    if not historical_sets: return None
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    counts = Counter(tuple(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in s).get(cat, 0) for cat in ['caliente', 'tibio', 'frio', 'congelado']) for s in historical_sets)
    return counts if counts else None

# --- Motores de Generación y Filtrado ---
def generar_combinaciones_con_restricciones(params):
    dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold = params
    valores = list(dist_prob.keys()); combinaciones = []; intentos = 0; max_intentos = n_comb * 400
    while len(combinaciones) < n_comb and intentos < max_intentos:
        intentos += 1
        seleccionados_str = random.sample(valores, n_sel); seleccionados = [int(n) for n in seleccionados_str]
        if not (sum_range[0] <= sum(seleccionados) <= sum_range[1]): continue
        if sum(1 for n in seleccionados if n % 2 == 0) not in parity_counts_allowed: continue
        nums = sorted(seleccionados); current_consecutive = 1; max_consecutive = 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1: current_consecutive += 1
            else: max_consecutive = max(max_consecutive, current_consecutive); current_consecutive = 1
        if max(max_consecutive, current_consecutive) > max_consecutive_allowed: continue
        freqs = [num_a_freq.get(str(val), 0) for val in seleccionados]; mean_freq = np.mean(freqs)
        if mean_freq == 0 or not (freq_cv_range[0] <= (np.std(freqs) / mean_freq) <= freq_cv_range[1]): continue
        suma_atrasos = sum(num_a_atraso.get(str(val), 0) for val in seleccionados); valor_especial = total_atraso + 40 - suma_atrasos
        if not (special_range[0] <= valor_especial <= special_range[1]): continue
        if any(Counter(num_a_atraso.get(str(n), -1) for n in seleccionados)[int(a)] > l for a, l in restr_atraso.items()): continue
        if hist_combs and any(len(set(seleccionados).intersection(h)) > hist_similarity_threshold for h in hist_combs): continue
        combinaciones.append(tuple(sorted(seleccionados)))
    conteo = Counter(combinaciones)
    return sorted({c: (f, np.prod([dist_prob.get(str(v), 0) for v in c])) for c, f in conteo.items()}.items(), key=lambda x: -x[1][1])

def procesar_combinaciones(params_tuple, n_ejec):
    with ProcessPoolExecutor() as executor:
        return [future.result() for future in as_completed([executor.submit(generar_combinaciones_con_restricciones, params_tuple) for _ in range(n_ejec)])]

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    return [c for c in combinaciones if all(Counter(get_category(numero_a_atraso.get(str(n),-1), composicion_rules['ranges']) for n in c).get(cat,0)==cnt for cat,cnt in composicion_rules['counts'].items())]

def evaluar_individuo_deap(individuo_str, params):
    dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, hist_combs, total_atraso, special_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold = params
    individuo = [int(n) for n in individuo_str]
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    if not (sum_range[0] <= sum(individuo) <= sum_range[1]): return (0,)
    if sum(1 for n in individuo if n % 2 == 0) not in parity_counts_allowed: return (0,)
    nums = sorted(individuo); current_consecutive=1; max_consecutive=0
    for i in range(1, len(nums)):
        if nums[i]==nums[i-1]+1: current_consecutive+=1
        else: max_consecutive=max(max_consecutive, current_consecutive); current_consecutive=1
    if max(max_consecutive, current_consecutive) > max_consecutive_allowed: return (0,)
    freqs = [num_a_freq.get(str(val), 0) for val in individuo]; mean_freq = np.mean(freqs)
    if mean_freq==0 or not (freq_cv_range[0] <= (np.std(freqs) / mean_freq) <= freq_cv_range[1]): return (0,)
    if any(Counter(num_a_atraso.get(str(n),-1) for n in individuo)[int(a)] > l for a,l in restr_atraso.items()): return (0,)
    if hist_combs and any(len(set(individuo).intersection(h)) > hist_similarity_threshold for h in hist_combs): return (0,)
    suma_atrasos = sum(num_a_atraso.get(str(val), 0) for val in individuo)
    valor_especial = total_atraso + 40 - suma_atrasos
    if not (special_range[0] <= valor_especial <= special_range[1]): return (0,)
    return (np.prod([dist_prob.get(str(val), 0) for val in individuo]),)

def ejecutar_algoritmo_genetico_hibrido(initial_population, ga_params, backend_params):
    n_gen, n_pob, cxpb, mutpb, n_sel = ga_params; toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluar_individuo_deap, params=backend_params); toolbox.register("mate", tools.cxTwoPoint); toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1); toolbox.register("select", tools.selTournament, tournsize=3)
    population = [creator.Individual(str(item) for item in comb) for comb in (random.sample(initial_population, n_pob) if len(initial_population) > n_pob else initial_population)]
    if not population: return [], "La población inicial (de la Etapa 1) estaba vacía."
    hall_of_fame = tools.HallOfFame(20)
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, n_gen, stats=None, halloffame=hall_of_fame, verbose=False)
    if not hall_of_fame: return [], "El AG no pudo encontrar ninguna solución válida."
    return hall_of_fame.items, None

def seleccionar_salon_de_la_fama_diverso(hall_of_fame_items, num_a_seleccionar=5, umbral_similitud=3):
    if not hall_of_fame_items: return []
    portafolio_final = []; hall_of_fame_sets = [set(int(n) for n in ind) for ind in hall_of_fame_items]
    portafolio_final.append(hall_of_fame_sets[0])
    for candidato in hall_of_fame_sets[1:]:
        if len(portafolio_final) >= num_a_seleccionar: break
        if all(len(candidato.intersection(miembro_existente)) <= umbral_similitud for miembro_existente in portafolio_final):
            portafolio_final.append(candidato)
    return [sorted(list(comb)) for comb in portafolio_final]

# ----------------------- ESTRUCTURA PRINCIPAL DE LA APLICACIÓN -----------------------

st.set_page_config(layout="wide", page_title="Modelo Híbrido Homeostático")
st.title("Modelo Híbrido Homeostático (Exploración + Optimización)")
if 'suggested_composition' not in st.session_state: st.session_state.suggested_composition = None

st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2);
with col1:
    uploaded_file = st.file_uploader("Sube CSV ('Numero', 'Atraso', 'Frecuencia')", type="csv", key="data_uploader")
    df, num_a_atraso, num_a_freq, dist_prob, atrasos_disp, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    hist_file = st.file_uploader("Sube CSV con Historial de Combinaciones", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(hist_file)

n_selecciones = 6
if df is not None: st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso}")

st.header("2. Configuración de Filtros de Precisión")
restricciones_finales, composicion_rules, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold = {}, {}, (0, 999), [], 6, 6
special_calc_range, freq_cv_range = (0, 99999), (0.0, 999.9)
if df is not None:
    st.subheader("Filtros de Homeostasis (Etapa 1)")
    if historical_combinations_set:
        col_freq, col_spec = st.columns(2)
        with col_freq:
            with st.expander("Filtro por CV de Frecuencia (Largo Plazo)", expanded=True):
                stats_freq_cv = analyze_historical_frequency_cv(historical_combinations_set, num_a_freq)
                if stats_freq_cv:
                    st.info(f"Historial: CV Frec. varía de **{stats_freq_cv['min']:.2f}** a **{stats_freq_cv['max']:.2f}**.")
                    slider_min_cv, slider_max_cv = 0.0, 2.0; default_start_cv = max(slider_min_cv, stats_freq_cv['mean'] - stats_freq_cv['std']); default_end_cv = min(slider_max_cv, stats_freq_cv['mean'] + stats_freq_cv['std'])
                    freq_cv_range = st.slider("Rango de CV:", slider_min_cv, slider_max_cv, (default_start_cv, default_end_cv), format="%.2f", key="freq_cv_slider")
        with col_spec:
            with st.expander("Filtro de 'Cálculo Especial' (Corto Plazo)", expanded=True):
                stats_special = analyze_historical_special_calc(historical_combinations_set, total_atraso, num_a_atraso)
                if stats_special:
                    st.info(f"Historial: 'Cálculo Especial' varía de **{stats_special['min']}** a **{stats_special['max']}**.")
                    slider_min_special, slider_max_special = float(stats_special['min'] - 50), float(stats_special['max'] + 50)
                    default_start_special = max(slider_min_special, float(stats_special['mean'] - stats_special['std'])); default_end_special = min(slider_max_special, float(stats_special['mean'] + stats_special['std']))
                    special_calc_range = st.slider("Rango de Cálculo Especial:", slider_min_special, slider_max_special, (default_start_special, default_end_special), key="special_slider")

    st.subheader("Filtros de Estructura Interna (Etapa 1)")
    if historical_combinations_set:
        sum_stats, parity_stats, consecutive_stats = analyze_historical_structure(historical_combinations_set)
        col_sum, col_par, col_cons = st.columns(3)
        with col_sum:
            with st.expander("Suma de la Combinación", expanded=True):
                if sum_stats:
                    st.info(f"Historial: Suma varía de {sum_stats['min']} a {sum_stats['max']}."); default_sum = (float(sum_stats['mean'] - sum_stats['std']), float(sum_stats['mean'] + sum_stats['std']))
                    sum_range = st.slider("Rango de Suma:", float(sum_stats['min'] - 20), float(sum_stats['max'] + 20), default_sum)
        with col_par:
            with st.expander("Cantidad de Números Pares", expanded=True):
                if parity_stats:
                    options = sorted(list(parity_stats.keys())); st.info(f"Distribución histórica: {dict(parity_stats.most_common())}")
                    parity_counts_allowed = st.multiselect("Nº Pares Permitidos:", options, default=options)
        with col_cons:
            with st.expander("Máx. Números Consecutivos", expanded=True):
                if consecutive_stats:
                    st.info(f"Distribución histórica: {dict(consecutive_stats.most_common())}")
                    max_consecutive_allowed = st.number_input("Máximo de Consecutivos:", 1, n_selecciones, 2)
    
    st.subheader("Filtros Estratégicos Adicionales")
    with st.expander("Atraso Individual, Similitud (Etapa 1) y Composición (Opcional)"):
        st.write("**Filtro de Atrasos Individuales (Etapa 1)**"); selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", [str(a) for a in atrasos_disp], default=[str(a) for a in atrasos_disp]); cols_ui_atraso = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols_ui_atraso[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", 0, n_selecciones, atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}"); restricciones_finales[atraso_str] = limit
        st.write("**Umbral de Similitud Histórica (Etapa 1)**"); hist_similarity_threshold = st.slider("Máx. repetidos de sorteos pasados:", 0, 5, 2)
        st.write("**Filtro Opcional de Composición (Post-AG)**"); max_atraso = atraso_stats.get("max", 100)
        c1, c2 = st.columns(2)
        with c1: range_caliente = st.slider("Rango 'Caliente'", 0, max_atraso, (0, int(atraso_stats.get("p25", 5))), key="r_hot"); range_frio = st.slider("Rango 'Frío'", 0, max_atraso, (int(atraso_stats.get("p75", 15)), max_atraso - 1), key="r_cold")
        with c2: range_tibio = st.slider("Rango 'Tibio'", 0, max_atraso, (range_caliente[1] + 1, range_frio[0] -1), key="r_warm"); min_congelado = st.number_input("Mínimo 'Congelado'", value=max_atraso, key="r_icy")
        current_ranges = {'caliente': range_caliente, 'tibio': range_tibio, 'frio': range_frio, 'congelado': (min_congelado, 9999)}
        if historical_combinations_set:
            comp_analysis = analyze_historical_composition(historical_combinations_set, num_a_atraso, current_ranges)
            if comp_analysis:
                most_common = comp_analysis.most_common(1)[0][0]
                st.success(f"Recomendación Historial: {most_common[0]} Cal, {most_common[1]} Tib, {most_common[2]} Frí, {most_common[3]} Con");
                if st.button("Aplicar"): st.session_state.suggested_composition = most_common; st.rerun()
        suggested = st.session_state.suggested_composition; c3,c4,c5,c6 = st.columns(4)
        count_caliente = c3.number_input("Nº Calientes", 0, n_selecciones, suggested[0] if suggested else 2, key="c_hot"); count_tibio = c4.number_input("Nº Tibios", 0, n_selecciones, suggested[1] if suggested else 2, key="c_warm"); count_frio = c5.number_input("Nº Fríos", 0, n_selecciones, suggested[2] if suggested else 2, key="c_cold"); count_congelado = c6.number_input("Nº Congelados", 0, n_selecciones, suggested[3] if suggested else 0, key="c_icy")
        total_count_composition = count_caliente + count_tibio + count_frio + count_congelado
        if total_count_composition == n_selecciones: composicion_rules = {'ranges': current_ranges, 'counts': {'caliente': count_caliente, 'tibio': count_tibio, 'frio': count_frio, 'congelado': count_congelado}}
    with st.expander("Configurar Parámetros del Modelo Híbrido"):
        col_sim, col_ga = st.columns(2)
        with col_sim: st.subheader("Etapa 1: Simulación"); sim_n_comb=st.number_input("Combinaciones a Generar/Ejec.", 1000, value=50000); sim_n_ejec=st.number_input("Ejecuciones Concurrentes", 1, value=8)
        with col_ga: st.subheader("Etapa 2: AG"); ga_ngen=st.slider("Generaciones",10,500,50); ga_npob=st.slider("Tamaño Población AG",100,5000,1000); ga_cxpb=st.slider("Prob. Cruce",0.0,1.0,0.8); ga_mutpb=st.slider("Prob. Mutación",0.0,1.0,0.1)
else:
    st.info("Carga los archivos para configurar los filtros.")

st.header("3. Ejecutar Modelo Híbrido")
if df is not None:
    backend_params = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold)
    
    # NUEVO: Control para el portafolio final
    n_final_combinations = st.sidebar.number_input("Número de combinaciones del portafolio final:", min_value=1, max_value=20, value=5, key="n_final")

    if st.button("🚀 Ejecutar Modelo Híbrido (Simulación + AG)"):
        st.subheader("Resultados del Proceso Híbrido")
        with st.spinner("Etapa 1: Explorando masivamente el espacio de soluciones..."):
            start_time = time.time()
            params_sim = (dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range, sum_range, parity_counts_allowed, max_consecutive_allowed, hist_similarity_threshold)
            resultados_mc = procesar_combinaciones(params_sim, sim_n_ejec)
            st.info(f"Etapa 1 completada en {time.time() - start_time:.2f} s.")
        
        plausible_combinations = list(set(tuple(int(n) for n in c) for res in resultados_mc for c, _ in res))
        st.info(f"🧬 Caldo de Cultivo: **{len(plausible_combinations)}** combinaciones de élite únicas generadas.")

        if not plausible_combinations:
            st.warning("La Etapa 1 no encontró combinaciones plausibles con los filtros actuales. El AG no puede ejecutarse. Intenta relajar los filtros.")
        else:
            with st.spinner("Etapa 2: Optimizando el caldo de cultivo con Algoritmo Genético..."):
                ga_params_hibrido = (ga_ngen, ga_npob, ga_cxpb, ga_mutpb, n_selecciones)
                salon_de_la_fama, err_msg = ejecutar_algoritmo_genetico_hibrido(plausible_combinations, ga_params_hibrido, backend_params)
            
            if err_msg:
                st.error(f"Error en Etapa 2 (AG): {err_msg}")
            elif salon_de_la_fama:
                st.success("✨ ¡Optimización completada! Se ha generado un 'Salón de la Fama'.")
                portafolio_optimizado = seleccionar_salon_de_la_fama_diverso(salon_de_la_fama, num_a_seleccionar=n_final_combinations)
                st.subheader(f"Portafolio Estratégico de {len(portafolio_optimizado)} Combinaciones Óptimas")
                data = []
                for i, comb in enumerate(portafolio_optimizado):
                    freqs = [num_a_freq.get(str(v),0) for v in comb]
                    data.append({"Ranking": f"#{i+1}", "Combinación": " - ".join(map(str, comb)), "CV Frecuencia": np.std(freqs)/np.mean(freqs) if np.mean(freqs) > 0 else 0, "Cálculo Especial": total_atraso + 40 - sum(num_a_atraso.get(str(v),0) for v in comb), "Suma": sum(comb), "Pares": sum(1 for n in comb if n % 2 == 0)})
                df_results = pd.DataFrame(data)
                df_results['CV Frecuencia'] = df_results['CV Frecuencia'].map('{:,.2f}'.format)
                st.dataframe(df_results.set_index("Ranking"))
            else:
                st.warning("La Etapa 2 (AG) no pudo encontrar soluciones válidas a partir de la población inicial.")
else:
    st.warning("Carga los archivos de datos para ejecutar los algoritmos.")

st.sidebar.header("Guía del Modelo Híbrido"); st.sidebar.markdown("Este modelo implementa un potente flujo de trabajo de dos etapas para una precisión máxima:"); st.sidebar.markdown("**Etapa 1: Exploración (Simulación)**\n- Genera millones de combinaciones.\n- Aplica filtros de **homeostasis y estructura** para crear un \"caldo de cultivo\" de candidatos de élite."); st.sidebar.markdown("**Etapa 2: Optimización (AG)**\n- El AG se siembra con los candidatos de élite.\n- Se enfoca en **cruzar y mutar las mejores soluciones** para encontrar \"superestrellas\" y construir un portafolio de campeones."); 
