
import streamlit as st
import pulp
import pandas as pd

st.set_page_config(page_title="Optimización Avanzada de Portafolio", layout="wide")

st.title("🚀 Optimización Avanzada de Portafolio de Proyectos")
st.write("Edita los valores de los proyectos, activa o desactiva restricciones estratégicas de manera dinámica y evalúa escenarios óptimos.")

# --- 1. Datos iniciales de los proyectos ---
initial_data = {
    "Proyecto": [f"P{i}" for i in range(1, 13)],
    "Beneficio": [95, 75, 80, 60, 85, 65, 70, 90, 88, 42, 105, 48],
    "Inversión (M COP)": [80, 55, 70, 45, 60, 40, 50, 75, 65, 30, 85, 35],
    "Horas Requeridas": [500, 350, 250, 400, 300, 350, 200, 450, 380, 220, 520, 180]
}
df_initial = pd.DataFrame(initial_data)

# --- Panel de Edición de Proyectos ---
st.subheader("📊 1. Configuración de Inversiones, Beneficios y Horas por Proyecto")
st.write("Modifica los valores directamente en la tabla a continuación:")
df_edited = st.data_editor(df_initial, num_rows="fixed", use_container_width=True, key="editor_proyectos")

# Extraer valores modificados
beneficios = df_edited["Beneficio"].tolist()
costos = df_edited["Inversión (M COP)"].tolist()
horas = df_edited["Horas Requeridas"].tolist()
n_proyectos = len(df_edited)

# --- 2. Panel Lateral de Configuración de Recursos y Condiciones ---
st.sidebar.header("⚙️ Configuración del Escenario")

st.sidebar.subheader("Límites de Recursos")
presupuesto_max = st.sidebar.number_input("Presupuesto máximo (M COP)", min_value=1, max_value=1000, value=300)
horas_max = st.sidebar.number_input("Disponibilidad de horas", min_value=1, max_value=5000, value=1800)
max_proyectos = st.sidebar.slider("Número máximo de proyectos", 1, 12, 6)

st.sidebar.subheader("Condiciones Estratégicas")
cond_8_1 = st.sidebar.checkbox("Si se selecciona P8, también P1 (x8 <= x1)", value=True)
cond_3_7_excl = st.sidebar.checkbox("P3 y P7 no simultáneos (x3 + x7 <= 1)", value=True)
cond_3_7_almenos = st.sidebar.checkbox("Al menos uno de P3 o P7 (x3 + x7 >= 1)", value=True)
cond_9_2 = st.sidebar.checkbox("Si se selecciona P9, también P2 (x9 <= x2)", value=True)
cond_4_11_excl = st.sidebar.checkbox("P4 y P11 no simultáneos (x4 + x11 <= 1)", value=True)
cond_1_2_5_almenos = st.sidebar.checkbox("Al menos uno de P1, P2 o P5 (x1 + x2 + x5 >= 1)", value=True)

# --- 3. Ejecutar Optimización ---
if st.button("▶️ Ejecutar Optimización de Portafolio"):
    # Definir el problema de maximización
    prob = pulp.LpProblem("Opt_Portafolio", pulp.LpMaximize)
    
    # Variables de decisión binarias
    x = {i: pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(1, n_proyectos + 1)}
    
    # Función objetivo
    prob += pulp.lpSum(beneficios[i-1] * x[i] for i in range(1, n_proyectos + 1)), "Beneficio_Total"
    
    # Restricción de recursos
    prob += pulp.lpSum(costos[i-1] * x[i] for i in range(1, n_proyectos + 1)) <= presupuesto_max, "Presupuesto"
    prob += pulp.lpSum(horas[i-1] * x[i] for i in range(1, n_proyectos + 1)) <= horas_max, "Horas"
    prob += pulp.lpSum(x[i] for i in range(1, n_proyectos + 1)) <= max_proyectos, "Max_Proyectos"
    
    # Restricciones Lógicas Condicionales según los Checkboxes
    if cond_8_1:
        prob += x[8] - x[1] <= 0, "Condicion_P8_P1"
    if cond_3_7_excl:
        prob += x[3] + x[7] <= 1, "Excluyentes_P3_P7"
    if cond_3_7_almenos:
        prob += x[3] + x[7] >= 1, "Al_menos_P3_P7"
    if cond_9_2:
        prob += x[9] - x[2] <= 0, "Condicion_P9_P2"
    if cond_4_11_excl:
        prob += x[4] + x[11] <= 1, "Excluyentes_P4_P11"
    if cond_1_2_5_almenos:
        prob += x[1] + x[2] + x[5] >= 1, "Al_menos_P1_P2_P5"
        
    # Resolver
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == "Optimal":
        st.success("¡Solución Óptima Encontrada!")
        
        beneficio_total = pulp.value(prob.objective)
        presupuesto_usado = sum(costos[i-1] * pulp.value(x[i]) for i in range(1, n_proyectos + 1))
        horas_usadas = sum(horas[i-1] * pulp.value(x[i]) for i in range(1, n_proyectos + 1))
        seleccionados_count = sum(pulp.value(x[i]) for i in range(1, n_proyectos + 1))
        
        # --- Métricas Principales ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Beneficio Total", f"{beneficio_total:.1f} pts")
        col2.metric("Presupuesto Usado", f"{presupuesto_usado:.1f} M", f"Límite: {presupuesto_max} M")
        col3.metric("Horas Usadas", f"{horas_usadas:.1f} hrs", f"Límite: {horas_max} hrs")
        col4.metric("Proyectos Elegidos", f"{int(seleccionados_count)} / {max_proyectos}")
        
        # --- Comparación Gráfica de Recursos ---
        st.subheader("📊 Análisis de Consumo de Recursos")
        col_bar1, col_bar2 = st.columns(2)
        
        with col_bar1:
            st.write(f"**Uso del Presupuesto:** {presupuesto_usado:.1f} de {presupuesto_max} M COP")
            st.progress(min(float(presupuesto_usado / presupuesto_max), 1.0))
            
        with col_bar2:
            st.write(f"**Uso de Horas:** {horas_usadas:.1f} de {horas_max} hrs")
            st.progress(min(float(horas_usadas / horas_max), 1.0))
            
        # --- Detalle de Resultados ---
        st.subheader("📌 Portafolio Óptimo Seleccionado")
        res_rows = []
        for i in range(1, n_proyectos + 1):
            seleccionado = pulp.value(x[i]) == 1.0
            res_rows.append({
                "Proyecto": f"P{i}",
                "Seleccionado": "✅ SÍ" if seleccionado else "❌ NO",
                "Beneficio": beneficios[i-1],
                "Inversión (M COP)": costos[i-1],
                "Horas Requeridas": horas[i-1]
            })
        
        df_res = pd.DataFrame(res_rows)
        st.dataframe(df_res.style.applymap(
            lambda val: 'background-color: #d4edda' if val == '✅ SÍ' else '',
            subset=['Seleccionado']
        ), use_container_width=True)
    else:
        st.error("❌ No se encontró una combinación viable de proyectos con las restricciones seleccionadas. Intenta aumentar el presupuesto, las horas disponibles o deshabilitar algunas condiciones estratégicas.")
