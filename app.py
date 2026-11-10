import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt

st.set_page_config(layout='wide', page_title='CapitalGestión S.A.S. - Optimización')

# --- DATOS INICIALES DEL PORTAFOLIO ---
proyectos_iniciales = [
    {"ID": "P1", "Nombre": "Automatización", "Beneficio": 95.0, "Inversion": 80.0, "Horas": 500.0},
    {"ID": "P2", "Nombre": "Analítica de datos", "Beneficio": 75.0, "Inversion": 55.0, "Horas": 350.0},
    {"ID": "P3", "Nombre": "Energía solar", "Beneficio": 80.0, "Inversion": 70.0, "Horas": 250.0},
    {"ID": "P4", "Nombre": "Infraestructura", "Beneficio": 60.0, "Inversion": 45.0, "Horas": 400.0},
    {"ID": "P5", "Nombre": "Mantenimiento predictivo", "Beneficio": 85.0, "Inversion": 60.0, "Horas": 300.0},
    {"ID": "P6", "Nombre": "Optimización logística", "Beneficio": 65.0, "Inversion": 40.0, "Horas": 350.0},
    {"ID": "P7", "Nombre": "Reutilización de agua", "Beneficio": 70.0, "Inversion": 50.0, "Horas": 200.0},
    {"ID": "P8", "Nombre": "Robotización", "Beneficio": 90.0, "Inversion": 75.0, "Horas": 450.0},
    {"ID": "P9", "Nombre": "Comercio electrónico", "Beneficio": 88.0, "Inversion": 65.0, "Horas": 380.0},
    {"ID": "P10", "Nombre": "Capacitación tecnológica", "Beneficio": 42.0, "Inversion": 30.0, "Horas": 220.0},
    {"ID": "P11", "Nombre": "Expansión comercial", "Beneficio": 105.0, "Inversion": 85.0, "Horas": 520.0},
    {"ID": "P12", "Nombre": "Gestión energética", "Beneficio": 48.0, "Inversion": 35.0, "Horas": 180.0}
]

st.title("CapitalGestión S.A.S.")
st.subheader("Optimización de Portafolio de Inversiones - Ingeniería Administrativa")
st.write("Esta herramienta permite simular, optimizar y analizar el portafolio de proyectos bajo diferentes escenarios y políticas estratégicas utilizando programación entera binaria.")

st.header("1. Parámetros del Portafolio y Datos de Proyectos")
st.write("Modifique los valores de la tabla directamente si desea evaluar variaciones en los proyectos individuales:")

df_init = pd.DataFrame(proyectos_iniciales)
edited_df = st.data_editor(df_init, num_rows='fixed', use_container_width=True, key='editor_proyectos')
proyectos_actuales = edited_df.to_dict('records')

# Entradas del modelo
col1, col2, col3 = st.columns(3)
with col1:
    presupuesto_b = st.number_input("Presupuesto Máximo (B) - Millones COP", value=300.0, step=10.0)
with col2:
    horas_h = st.number_input("Horas de Trabajo Disponibles (H)", value=1800.0, step=50.0)
with col3:
    max_proyectos_m = st.number_input("Límite Máximo de Proyectos (M)", value=6, step=1)

st.header("2. Políticas y Restricciones Estratégicas")
r_dep = st.checkbox("P8 depende de P1 (Robotización requiere Automatización: x8 - x1 <= 0)", value=True)
r_mut_1 = st.checkbox("P3 y P7 son mutuamente excluyentes (x3 + x7 <= 1)", value=True)
r_oblig = st.checkbox("Debe seleccionarse al menos uno entre P3 o P7 (x3 + x7 >= 1)", value=True)
r_ecommerce = st.checkbox("P9 depende de P2 (Comercio electrónico requiere Analítica de datos: x9 - x2 <= 0)", value=True)
r_mut_2 = st.checkbox("P4 y P11 son mutuamente excluyentes (x4 + x11 <= 1)", value=True)
r_min_grupo = st.checkbox("Debe seleccionarse al menos uno entre P1, P2 o P5 (x1 + x2 + x5 >= 1)", value=True)

# Función para construir y resolver el modelo con PuLP
def resolver_modelo(proy_list, B, H, M, act_dep, act_mut_1, act_oblig, act_ecom, act_mut_2, act_grupo):
    prob = pulp.LpProblem("Portafolio_Inversiones", pulp.LpMaximize)
    x = {p['ID']: pulp.LpVariable(f"x_{p['ID']}", cat=pulp.LpBinary) for p in proy_list}

    prob += pulp.lpSum(p['Beneficio'] * x[p['ID']] for p in proy_list), "Z_Beneficio"
    prob += pulp.lpSum(p['Inversion'] * x[p['ID']] for p in proy_list) <= B, "Limite_Presupuesto"
    prob += pulp.lpSum(p['Horas'] * x[p['ID']] for p in proy_list) <= H, "Limite_Horas"
    prob += pulp.lpSum(x[p['ID']] for p in proy_list) <= M, "Limite_Max_Proyectos"

    if act_dep and 'P8' in x and 'P1' in x:
        prob += x['P8'] - x['P1'] <= 0, "Politica_P8_P1"
    if act_mut_1 and 'P3' in x and 'P7' in x:
        prob += x['P3'] + x['P7'] <= 1, "Politica_MutEx_P3_P7"
    if act_oblig and 'P3' in x and 'P7' in x:
        prob += x['P3'] + x['P7'] >= 1, "Politica_Oblig_P3_P7"
    if act_ecom and 'P9' in x and 'P2' in x:
        prob += x['P9'] - x['P2'] <= 0, "Politica_P9_P2"
    if act_mut_2 and 'P4' in x and 'P11' in x:
        prob += x['P4'] + x['P11'] <= 1, "Politica_MutEx_P4_P11"
    if act_grupo and 'P1' in x and 'P2' in x and 'P5' in x:
        prob += x['P1'] + x['P2'] + x['P5'] >= 1, "Politica_Min_Grupo"
        if act_oblig ['p8'] >= 1: "Restriccion_adicc_P8 Obligatoriamente"



    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    return prob, status, x

if st.button("Optimizar Modelo Personalizado", type="primary"):
    st.header("3. Resultados de la Solución Personalizada")
    prob, status, vars_x = resolver_modelo(proyectos_actuales, presupuesto_b, horas_h, max_proyectos_m, r_dep, r_mut_1, r_oblig, r_ecommerce, r_mut_2, r_min_grupo)
    st.write(f"**Estado de la Solución del Solucionador:** `{status}`")

    if status == 'Optimal':
        p_dict = {p['ID']: p for p in proyectos_actuales}
        seleccionados = [p['ID'] for p in proyectos_actuales if vars_x[p['ID']].varValue > 0.5]
        total_z_calc = sum(p_dict[i]['Beneficio'] for i in seleccionados)
        total_inv = sum(p_dict[i]['Inversion'] for i in seleccionados)
        total_h = sum(p_dict[i]['Horas'] for i in seleccionados)
        num_p = len(seleccionados)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Beneficio Total (Z)", f"{total_z_calc:.2f}")
        col_m2.metric("Presupuesto Utilizado", f"{total_inv:.2f} / {presupuesto_b:.2f}")
        col_m3.metric("Horas Utilizadas", f"{total_h:.2f} / {horas_h:.2f}")
        col_m4.metric("Proyectos Seleccionados", f"{num_p} / {max_proyectos_m}")

        st.subheader("Detalle de Proyectos Seleccionados")
        df_sel = edited_df[edited_df['ID'].isin(seleccionados)]
        st.dataframe(df_sel, use_container_width=True)

        st.subheader("Tabla de Verificación y Validación de Políticas")
        verif_datos = []
        verif_datos.append({"Restricción": "Presupuesto Utilizado (<= B)", "Límite / Condición": f"<= {presupuesto_b}", "Valor Evaluado": total_inv, "¿Cumple?": "Sí" if total_inv <= presupuesto_b else "No"})
        verif_datos.append({"Restricción": "Horas de Trabajo (<= H)", "Límite / Condición": f"<= {horas_h}", "Valor Evaluado": total_h, "¿Cumple?": "Sí" if total_h <= horas_h else "No"})
        verif_datos.append({"Restricción": "Límite Proyectos (<= M)", "Límite / Condición": f"<= {max_proyectos_m}", "Valor Evaluado": num_p, "¿Cumple?": "Sí" if num_p <= max_proyectos_m else "No"})

        if r_dep:
            val_dep = int('P8' in seleccionados) - int('P1' in seleccionados)
            verif_datos.append({"Restricción": "Robotización requiere Automatización (x8 - x1 <= 0)", "Límite / Condición": "<= 0", "Valor Evaluado": val_dep, "¿Cumple?": "Sí" if val_dep <= 0 else "No"})
        if r_mut_1:
            val_mut = int('P3' in seleccionados) + int('P7' in seleccionados)
            verif_datos.append({"Restricción": "P3 y P7 mutuamente excluyentes (x3 + x7 <= 1)", "Límite / Condición": "<= 1", "Valor Evaluado": val_mut, "¿Cumple?": "Sí" if val_mut <= 1 else "No"})
        if r_oblig:
            val_obl = int('P3' in seleccionados) + int('P7' in seleccionados)
            verif_datos.append({"Restricción": "Debe seleccionarse al menos uno entre P3 o P7 (x3 + x7 >= 1)", "Límite / Condición": ">= 1", "Valor Evaluado": val_obl, "¿Cumple?": "Sí" if val_obl >= 1 else "No"})
        if r_ecommerce:
            val_ecom = int('P9' in seleccionados) - int('P2' in seleccionados)
            verif_datos.append({"Restricción": "E-commerce requiere Analítica de datos (x9 - x2 <= 0)", "Límite / Condición": "<= 0", "Valor Evaluado": val_ecom, "¿Cumple?": "Sí" if val_ecom <= 0 else "No"})
        if r_mut_2:
            val_mut2 = int('P4' in seleccionados) + int('P11' in seleccionados)
            verif_datos.append({"Restricción": "P4 y P11 mutuamente excluyentes (x4 + x11 <= 1)", "Límite / Condición": "<= 1", "Valor Evaluado": val_mut2, "¿Cumple?": "Sí" if val_mut2 <= 1 else "No"})
        if r_min_grupo:
            val_g = int('P1' in seleccionados) + int('P2' in seleccionados) + int('P5' in seleccionados)
            verif_datos.append({"Restricción": "Al menos uno del grupo P1, P2 o P5 (x1 + x2 + x5 >= 1)", "Límite / Condición": ">= 1", "Valor Evaluado": val_g, "¿Cumple?": "Sí" if val_g >= 1 else "No"})

        val_solver_z = pulp.value(prob.objective)
        verif_datos.append({"Restricción": "Conciliación de Beneficio (Z calculado vs Solucionador)", "Límite / Condición": f"Igual a {val_solver_z:.2f}", "Valor Evaluado": total_z_calc, "¿Cumple?": "Sí" if abs(total_z_calc - val_solver_z) < 1e-4 else "No"})
        st.dataframe(pd.DataFrame(verif_datos), use_container_width=True)

        st.subheader("Análisis Gráfico de Utilización de Recursos")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(["Presupuesto"], [total_inv], color='skyblue', label='Utilizado')
        ax[0].axhline(presupuesto_b, color='red', linestyle='--', label='Disponible')
        ax[0].set_ylabel('Millones COP')
        ax[0].set_ylim(0, max(presupuesto_b, total_inv) * 1.25)
        ax[0].legend()

        ax[1].bar(["Horas de Trabajo"], [total_h], color='lightgreen', label='Utilizado')
        ax[1].axhline(horas_h, color='red', linestyle='--', label='Disponible')
        ax[1].set_ylabel('Horas')
        ax[1].set_ylim(0, max(horas_h, total_h) * 1.25)
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.error(f"El modelo no se puede resolver en este estado. Código de error: {status}.")

st.header("3. Tabla Comparativa de Escenarios")
st.write("Se ejecutan de manera independiente los escenarios predefinidos aplicando las modificaciones exactas:")

def obtener_metricas_escenario(proyectos_modificados, B, H, M):
    p_dict_loc = {p['ID']: p for p in proyectos_modificados}
    _, status_l, x_l = resolver_modelo(proyectos_modificados, B, H, M, r_dep, r_mut_1, r_oblig, r_ecommerce, r_mut_2, r_min_grupo)
    if status_l == 'Optimal':
        sel = [p['ID'] for p in proyectos_modificados if x_l[p['ID']].varValue > 0.5]
        z_v = sum(p_dict_loc[i]['Beneficio'] for i in sel)
        inv_v = sum(p_dict_loc[i]['Inversion'] for i in sel)
        h_v = sum(p_dict_loc[i]['Horas'] for i in sel)
        return status_l, sel, z_v, inv_v, h_v, len(sel)
    return status_l, [], None, None, None, None

status_base, sel_base, z_base, inv_base, h_base, cant_base = obtener_metricas_escenario(proyectos_actuales, 300.0, 1800.0, 6)
status_e1, sel_e1, z_e1, inv_e1, h_e1, cant_e1 = obtener_metricas_escenario(proyectos_actuales, 230.0, 1800.0, 6)

proy_e2 = []
for p in proyectos_actuales:
    copia = p.copy()
    if copia['ID'] in ['P1', 'P2', 'P8']:
        copia['Beneficio'] = copia['Beneficio'] * 1.30
    proy_e2.append(copia)
status_e2, sel_e2, z_e2, inv_e2, h_e2, cant_e2 = obtener_metricas_escenario(proy_e2, 300.0, 1800.0, 6)

status_e3, sel_e3, z_e3, inv_e3, h_e3, cant_e3 = obtener_metricas_escenario(proyectos_actuales, 380.0, 2200.0, 6)
status_e4, sel_e4, z_e4, inv_e4, h_e4, cant_e4 = obtener_metricas_escenario(proyectos_actuales, 150.0, 1000.0, 6)

filas_resumen = []
def formatear_linea(nombre, status, sel, z, inv, h, cant, base_z):
    if status == 'Optimal':
        diff = f"{z - base_z:+.2f}" if base_z is not None else "0.00"
        return [nombre, status, ", ".join(sel), round(z, 2), round(inv, 2), round(h, 2), cant, diff]
    return [nombre, status, "-", "-", "-", "-", "-", "-"]

filas_resumen.append(formatear_linea("Base (300, 1800, 6)", status_base, sel_base, z_base, inv_base, h_base, cant_base, z_base))
filas_resumen.append(formatear_linea("E1 (B = 230)", status_e1, sel_e1, z_e1, inv_e1, h_e1, cant_e1, z_base))
filas_resumen.append(formatear_linea("E2 (Precios P1,P2,P8 * 1.30)", status_e2, sel_e2, z_e2, inv_e2, h_e2, cant_e2, z_base))
filas_resumen.append(formatear_linea("E3 (B = 380, H = 2200)", status_e3, sel_e3, z_e3, inv_e3, h_e3, cant_e3, z_base))
filas_resumen.append(formatear_linea("E4 (Prueba Validación: B=150, H=1000)", status_e4, sel_e4, z_e4, inv_e4, h_e4, cant_e4, z_base))

df_comparativa = pd.DataFrame(filas_resumen, columns=["Escenario", "Estado", "Proyectos", "Beneficio (Z)", "Inversión Utilizada", "Horas Utilizadas", "Cantidad", "Cambio Z vs Base"])
st.dataframe(df_comparativa, use_container_width=True)

st.subheader("Análisis Automático de Cambios de Escenario")
if status_base == 'Optimal':
    base_set = set(sel_base)
    dict_scens = {
        "E1 (B = 230)": (status_e1, sel_e1, z_e1),
        "E2 (Precios P1,P2,P8 * 1.30)": (status_e2, sel_e2, z_e2),
        "E3 (B = 380, H = 2200)": (status_e3, sel_e3, z_e3),
        "E4 (Prueba Validación: B=150, H=1000)": (status_e4, sel_e4, z_e4)
    }
    for nombre, (st_l, sel_l, z_l) in dict_scens.items():
        st.markdown(f"### Análisis de {nombre}")
        if st_l == 'Optimal':
            sel_set = set(sel_l)
            entran = list(sel_set - base_set)
            salen = list(base_set - sel_set)
            cambio_z = z_l - z_base
            st.write(f"* **Estado**: {st_l}")
            st.write(f"* **Beneficio Total (Z)**: {z_l:.2f} (Variación respecto al Base: {cambio_z:+.2f})")
            st.write(f"* **Proyectos que entran**: {', '.join(entran) if entran else 'Ninguno'}")
            st.write(f"* **Proyectos que salen**: {', '.join(salen) if salen else 'Ninguno'}")
        elif st_l == 'Infeasible':
            st.error(f"* **Estado**: {st_l}. El escenario es Infactible bajo las restricciones del modelo.")
        else:
            st.warning(f"* **Estado**: {st_l}. No se obtuvo solución óptima.")
else:
    st.warning("El escenario Base no es óptimo, no es posible comparar.")
