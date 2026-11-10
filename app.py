from copy import deepcopy
import json

import pandas as pd
import pulp
import streamlit as st


st.set_page_config(
    page_title="CapitalGestión | Optimización de portafolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.35rem;font-weight:800;color:#173B67;margin-bottom:.2rem;}
    .sub-title {font-size:1.05rem;color:#52606D;margin-bottom:1.4rem;}
    div[data-testid="stMetric"] {background:#F5F8FC;border-radius:12px;padding:14px;
      border-left:5px solid #2F80ED;box-shadow:0 2px 8px rgba(20,45,80,.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">💼 Optimización del portafolio de inversiones</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">CapitalGestión S.A.S. · Programación entera binaria para maximizar el beneficio esperado</div>',
    unsafe_allow_html=True,
)


NAMES = [
    "Automatización",
    "Analítica de datos",
    "Energía solar",
    "Infraestructura",
    "Mantenimiento predictivo",
    "Optimización logística",
    "Reutilización de agua",
    "Robotización",
    "Comercio electrónico",
    "Capacitación tecnológica",
    "Expansión comercial",
    "Gestión energética",
]

ORIGINAL_DATA = pd.DataFrame(
    {
        "Proyecto": [f"P{i}" for i in range(1, 13)],
        "Nombre": NAMES,
        "Inversión (M COP)": [80, 55, 70, 45, 60, 40, 50, 75, 65, 30, 85, 35],
        "Beneficio (M COP)": [95, 75, 80, 60, 85, 65, 70, 90, 88, 42, 105, 48],
        "Horas": [500, 350, 250, 400, 300, 350, 200, 450, 380, 220, 520, 180],
    }
)

POLICY_LABELS = [
    "Si se selecciona P8, también debe seleccionarse P1",
    "P3 y P7 no pueden seleccionarse simultáneamente",
    "Debe seleccionarse al menos uno de P3 o P7",
    "Si se selecciona P9, también debe seleccionarse P2",
    "P4 y P11 son incompatibles",
    "Debe seleccionarse al menos uno de P1, P2 o P5",
]

STATUS_ES = {
    "Optimal": "Óptimo",
    "Infeasible": "Infactible",
    "Unbounded": "No acotado",
    "Not Solved": "No resuelto",
    "Undefined": "Indefinido",
}


def validate_inputs(data, budget, available_hours, maximum_projects):
    numeric_columns = ["Inversión (M COP)", "Beneficio (M COP)", "Horas"]
    if len(data) != 12:
        raise ValueError("La tabla debe conservar exactamente los 12 proyectos.")
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"La columna {column} solo puede contener valores numéricos no negativos.")
    if budget < 0 or available_hours < 0 or maximum_projects < 0:
        raise ValueError("Los límites de recursos no pueden ser negativos.")


def solve_portfolio(data, budget, available_hours, maximum_projects, policies):
    validate_inputs(data, budget, available_hours, maximum_projects)
    benefits = pd.to_numeric(data["Beneficio (M COP)"]).astype(float).tolist()
    investments = pd.to_numeric(data["Inversión (M COP)"]).astype(float).tolist()
    hours = pd.to_numeric(data["Horas"]).astype(float).tolist()

    problem = pulp.LpProblem("CapitalGestion_Portafolio", pulp.LpMaximize)
    x = {
        i: pulp.LpVariable(
            f"x_{i + 1}", lowBound=0, upBound=1, cat=pulp.LpBinary
        )
        for i in range(12)
    }

    problem += pulp.lpSum(benefits[i] * x[i] for i in range(12)), "Beneficio_total"
    problem += pulp.lpSum(investments[i] * x[i] for i in range(12)) <= budget, "Presupuesto"
    problem += pulp.lpSum(hours[i] * x[i] for i in range(12)) <= available_hours, "Horas"
    problem += pulp.lpSum(x[i] for i in range(12)) <= maximum_projects, "Maximo_proyectos"

    if policies[0]:
        problem += x[7] - x[0] <= 0, "P8_requiere_P1"
    if policies[1]:
        problem += x[2] + x[6] <= 1, "P3_P7_no_simultaneos"
    if policies[2]:
        problem += x[2] + x[6] >= 1, "Al_menos_P3_o_P7"
    if policies[3]:
        problem += x[8] - x[1] <= 0, "P9_requiere_P2"
    if policies[4]:
        problem += x[3] + x[10] <= 1, "P4_P11_incompatibles"
    if policies[5]:
        problem += x[0] + x[1] + x[4] >= 1, "Al_menos_P1_P2_o_P5"

    try:
        status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    except Exception as exc:
        return {
            "estado": "Error del solucionador",
            "estado_original": type(exc).__name__,
            "mensaje": str(exc),
        }

    status_original = pulp.LpStatus.get(status_code, "Otro")
    result = {
        "estado": STATUS_ES.get(status_original, f"Otro: {status_original}"),
        "estado_original": status_original,
        "mensaje": f"Estado informado por CBC: {status_original}",
    }
    if status_original != "Optimal":
        return result

    values = [int(round(float(pulp.value(x[i])))) for i in range(12)]
    selected_indices = [i for i, value in enumerate(values) if value == 1]
    total_benefit = sum(benefits[i] * values[i] for i in range(12))
    used_investment = sum(investments[i] * values[i] for i in range(12))
    used_hours = sum(hours[i] * values[i] for i in range(12))
    count = sum(values)
    tolerance = 1e-6

    checks = [
        {
            "Restricción": "Presupuesto",
            "Lado izquierdo": used_investment,
            "Signo": "≤",
            "Límite": budget,
            "Cumple": used_investment <= budget + tolerance,
        },
        {
            "Restricción": "Horas disponibles",
            "Lado izquierdo": used_hours,
            "Signo": "≤",
            "Límite": available_hours,
            "Cumple": used_hours <= available_hours + tolerance,
        },
        {
            "Restricción": "Máximo de proyectos",
            "Lado izquierdo": count,
            "Signo": "≤",
            "Límite": maximum_projects,
            "Cumple": count <= maximum_projects,
        },
    ]

    policy_checks = [
        ("P8 requiere P1", values[7] - values[0], "≤", 0, values[7] <= values[0]),
        ("P3 y P7 no simultáneos", values[2] + values[6], "≤", 1, values[2] + values[6] <= 1),
        ("Al menos P3 o P7", values[2] + values[6], "≥", 1, values[2] + values[6] >= 1),
        ("P9 requiere P2", values[8] - values[1], "≤", 0, values[8] <= values[1]),
        ("P4 y P11 incompatibles", values[3] + values[10], "≤", 1, values[3] + values[10] <= 1),
        ("Al menos P1, P2 o P5", values[0] + values[1] + values[4], "≥", 1, values[0] + values[1] + values[4] >= 1),
    ]
    for active, check in zip(policies, policy_checks):
        if active:
            name, lhs, sign, limit, complies = check
            checks.append(
                {
                    "Restricción": name,
                    "Lado izquierdo": lhs,
                    "Signo": sign,
                    "Límite": limit,
                    "Cumple": complies,
                }
            )

    solver_objective = float(pulp.value(problem.objective))
    objective_verified = abs(solver_objective - total_benefit) <= tolerance
    all_checks = all(row["Cumple"] for row in checks) and objective_verified

    result.update(
        {
            "x": values,
            "seleccion": [data.iloc[i]["Proyecto"] for i in selected_indices],
            "beneficio": total_benefit,
            "inversion": used_investment,
            "horas": used_hours,
            "cantidad": count,
            "comprobacion": checks,
            "objetivo_solucionador": solver_objective,
            "objetivo_recalculado": total_benefit,
            "objetivo_verificado": objective_verified,
            "todo_cumple": all_checks,
        }
    )
    return result


def scenario_definitions():
    base = {
        "datos": ORIGINAL_DATA.copy(deep=True),
        "presupuesto": 300.0,
        "horas_max": 1800.0,
        "max_proyectos": 6,
        "politicas": [True] * 6,
    }
    scenarios = {
        "Base": deepcopy(base),
        "1. Presupuesto reducido": deepcopy(base),
        "2. Prioridad estratégica": deepcopy(base),
        "3. Aumento de recursos": deepcopy(base),
        "Prueba adicional: presupuesto cero": deepcopy(base),
    }
    scenarios["1. Presupuesto reducido"]["presupuesto"] = 230.0
    priority_data = scenarios["2. Prioridad estratégica"]["datos"]
    priority_data["Beneficio (M COP)"] = priority_data[
        "Beneficio (M COP)"
    ].astype(float)
    priority_data.loc[
        priority_data["Proyecto"].isin(["P1", "P2", "P8"]),
        "Beneficio (M COP)",
    ] *= 1.30
    scenarios["3. Aumento de recursos"].update(
        presupuesto=380.0, horas_max=2200.0
    )
    scenarios["Prueba adicional: presupuesto cero"]["presupuesto"] = 0.0
    return scenarios


def binding_conditions(result, limits):
    if result["estado"] != "Óptimo":
        return "No aplica"
    conditions = []
    if abs(limits["presupuesto"] - result["inversion"]) <= 1e-6:
        conditions.append("presupuesto")
    if abs(limits["horas_max"] - result["horas"]) <= 1e-6:
        conditions.append("horas")
    if abs(limits["max_proyectos"] - result["cantidad"]) <= 1e-6:
        conditions.append("máximo de proyectos")
    conditions.append("exactamente uno de P3/P7")
    if "P8" in result["seleccion"]:
        conditions.append("P8 requiere P1")
    if "P9" in result["seleccion"]:
        conditions.append("P9 requiere P2")
    return ", ".join(conditions)


def show_result(result, data, limits, key):
    st.subheader(f"Estado del solucionador: {result['estado']}")
    if result["estado"] != "Óptimo":
        if result["estado"] == "Infactible":
            st.warning(
                "No existe una combinación que satisfaga simultáneamente todas las restricciones. "
                "No se muestran valores de decisión porque no serían una solución válida."
            )
        else:
            st.error(result.get("mensaje", "El solucionador no entregó una solución óptima."))
        return

    columns = st.columns(4)
    columns[0].metric("Beneficio total", f"{result['beneficio']:,.1f} M COP")
    columns[1].metric("Inversión utilizada", f"{result['inversion']:,.1f} M COP")
    columns[2].metric("Horas utilizadas", f"{result['horas']:,.0f}")
    columns[3].metric("Proyectos seleccionados", f"{result['cantidad']} de {limits['max_proyectos']}")

    selected_table = data.copy()
    selected_table["xᵢ"] = result["x"]
    selected_table["Decisión"] = selected_table["xᵢ"].map(
        {1: "Seleccionado", 0: "No seleccionado"}
    )
    st.subheader("Portafolio óptimo seleccionado")
    st.dataframe(
        selected_table[selected_table["xᵢ"] == 1],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Utilización de recursos")
    budget_percentage = (
        100 * result["inversion"] / limits["presupuesto"]
        if limits["presupuesto"] > 0
        else 0
    )
    hours_percentage = (
        100 * result["horas"] / limits["horas_max"]
        if limits["horas_max"] > 0
        else 0
    )
    count_percentage = (
        100 * result["cantidad"] / limits["max_proyectos"]
        if limits["max_proyectos"] > 0
        else 0
    )
    usage = pd.DataFrame(
        {
            "Recurso": ["Presupuesto", "Horas", "Número de proyectos"],
            "Utilización (%)": [budget_percentage, hours_percentage, count_percentage],
        }
    ).set_index("Recurso")
    st.bar_chart(usage)

    with st.expander("Verificación matemática", expanded=True):
        verification = pd.DataFrame(result["comprobacion"])
        verification["Cumple"] = verification["Cumple"].map(
            {True: "Sí", False: "No"}
        )
        st.dataframe(verification, hide_index=True, width="stretch")
        if result["todo_cumple"]:
            st.success(
                f"Todas las restricciones se cumplen. Función objetivo verificada: "
                f"{result['objetivo_solucionador']:.1f} = {result['objetivo_recalculado']:.1f}."
            )
        else:
            st.error("La solución no superó la verificación matemática.")

    st.download_button(
        "Descargar este resultado",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"resultado_{key}.json",
        mime="application/json",
        key=f"download_{key}",
    )


with st.sidebar:
    st.header("Configuración del modelo editable")
    st.caption("Estos controles se aplican a la pestaña Modelo editable.")
    budget_input = st.number_input(
        "Presupuesto máximo (M COP)", min_value=0.0, value=300.0, step=10.0
    )
    hours_input = st.number_input(
        "Horas disponibles", min_value=0.0, value=1800.0, step=50.0
    )
    maximum_input = st.number_input(
        "Máximo de proyectos", min_value=0, max_value=12, value=6, step=1
    )
    st.subheader("Políticas estratégicas")
    policy_inputs = [
        st.checkbox(label, value=True, key=f"policy_{index}")
        for index, label in enumerate(POLICY_LABELS)
    ]


editable_tab, scenarios_tab, formulation_tab = st.tabs(
    ["🧮 Modelo editable", "📊 Escenarios obligatorios", "📘 Formulación matemática"]
)

with editable_tab:
    st.write(
        "Modifique las inversiones, beneficios y horas directamente en la tabla. "
        "El resultado se calcula nuevamente con el solucionador al pulsar el botón."
    )
    edited_data = st.data_editor(
        ORIGINAL_DATA,
        disabled=["Proyecto", "Nombre"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            column: st.column_config.NumberColumn(min_value=0.0, step=1.0)
            for column in ["Inversión (M COP)", "Beneficio (M COP)", "Horas"]
        },
        key="project_editor",
    )
    if st.button("Ejecutar optimización", type="primary", key="run_editable"):
        try:
            limits = {
                "presupuesto": float(budget_input),
                "horas_max": float(hours_input),
                "max_proyectos": int(maximum_input),
            }
            st.session_state["editable_result"] = {
                "resultado": solve_portfolio(
                    edited_data,
                    limits["presupuesto"],
                    limits["horas_max"],
                    limits["max_proyectos"],
                    policy_inputs,
                ),
                "datos": edited_data.copy(),
                "limites": limits,
            }
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("editable_result", None)

    if "editable_result" in st.session_state:
        saved = st.session_state["editable_result"]
        st.caption(
            "Se muestra la última ejecución. Si cambia algún dato, pulse nuevamente Ejecutar optimización."
        )
        show_result(
            saved["resultado"], saved["datos"], saved["limites"], "editable"
        )

with scenarios_tab:
    st.write(
        "Cada escenario se resuelve de manera independiente a partir de los datos originales "
        "del profesor y con las seis políticas estratégicas activadas."
    )
    if st.button(
        "Calcular escenario base, tres obligatorios y prueba adicional",
        type="primary",
        key="run_scenarios",
    ):
        comparison_results = []
        for name, definition in scenario_definitions().items():
            result = solve_portfolio(
                definition["datos"],
                definition["presupuesto"],
                definition["horas_max"],
                definition["max_proyectos"],
                definition["politicas"],
            )
            comparison_results.append((name, definition, result))
        st.session_state["comparison_results"] = comparison_results

    if "comparison_results" in st.session_state:
        comparison_results = st.session_state["comparison_results"]
        rows = []
        base_benefit = None
        for name, definition, result in comparison_results:
            if name == "Base" and result["estado"] == "Óptimo":
                base_benefit = result["beneficio"]
            rows.append(
                {
                    "Escenario": name,
                    "Estado": result["estado"],
                    "Proyectos seleccionados": ", ".join(result.get("seleccion", [])),
                    "Beneficio (M COP)": result.get("beneficio"),
                    "Inversión (M COP)": result.get("inversion"),
                    "Horas": result.get("horas"),
                    "Cantidad": result.get("cantidad"),
                    "Condiciones limitantes o activas": binding_conditions(
                        result, definition
                    ),
                }
            )

        comparison = pd.DataFrame(rows)
        if base_benefit is not None:
            comparison["Cambio del beneficio frente a base"] = comparison[
                "Beneficio (M COP)"
            ].apply(
                lambda value: value - base_benefit if pd.notna(value) else None
            )

        st.subheader("Tabla comparativa")
        st.dataframe(comparison, hide_index=True, width="stretch")

        optimal_chart = comparison[comparison["Estado"] == "Óptimo"].set_index(
            "Escenario"
        )[["Beneficio (M COP)"]]
        st.subheader("Comparación del beneficio")
        st.bar_chart(optimal_chart)

        st.subheader("Análisis de cambios y recomendaciones")
        base_result = comparison_results[0][2]
        for name, definition, result in comparison_results[1:4]:
            if result["estado"] == "Óptimo" and base_result["estado"] == "Óptimo":
                change = result["beneficio"] - base_result["beneficio"]
                st.write(
                    f"**{name}:** selecciona {', '.join(result['seleccion'])}; "
                    f"el beneficio cambia en {change:+.1f} M COP frente al escenario base. "
                    f"Condiciones relevantes: {binding_conditions(result, definition)}."
                )
        extra_result = comparison_results[4][2]
        st.write(
            f"**Prueba adicional:** con presupuesto cero el estado es "
            f"**{extra_result['estado']}**. Esto comprueba que la aplicación identifica "
            "la infactibilidad y evita mostrar decisiones inválidas."
        )
        st.info(
            "Recomendación: use el escenario de recursos ampliados cuando la empresa pueda "
            "aumentar presupuesto y horas; si el presupuesto cae a 230 M COP, priorice el "
            "portafolio que conserva el mayor beneficio sin incumplir las políticas."
        )

        st.download_button(
            "Descargar comparación en CSV",
            data=comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name="comparacion_escenarios.csv",
            mime="text/csv",
            key="download_comparison",
        )

        for name, definition, result in comparison_results:
            with st.expander(f"Detalle: {name}"):
                show_result(result, definition["datos"], definition, name)

with formulation_tab:
    st.subheader("Conjuntos, índice, parámetros y unidades")
    st.markdown(
        r"""
        - Conjunto de proyectos: $I=\{1,2,\ldots,12\}$.
        - Índice: $i\in I$ representa cada proyecto.
        - $b_i$: beneficio esperado del proyecto $i$, en millones de COP.
        - $c_i$: inversión del proyecto $i$, en millones de COP.
        - $h_i$: horas de personal requeridas por el proyecto $i$.
        - $P$: presupuesto disponible; $H$: horas disponibles; $M$: número máximo de proyectos.

        **Variable de decisión**

        $$x_i=\begin{cases}1,&\text{si se selecciona el proyecto }i\\0,&\text{si no se selecciona}\end{cases}$$

        **Función objetivo**

        $$\max Z=\sum_{i=1}^{12} b_i x_i$$

        **Restricciones**

        $$\sum_{i=1}^{12}c_i x_i\le P$$
        $$\sum_{i=1}^{12}h_i x_i\le H$$
        $$\sum_{i=1}^{12}x_i\le M$$
        $$x_8-x_1\le0$$
        $$x_3+x_7\le1$$
        $$x_3+x_7\ge1$$
        $$x_9-x_2\le0$$
        $$x_4+x_{11}\le1$$
        $$x_1+x_2+x_5\ge1$$
        $$x_i\in\{0,1\}\quad\forall i\in I$$
        """
    )
    st.caption(
        "El modelo se resuelve en cada ejecución mediante PuLP y CBC; las respuestas no están precalculadas."
    )

st.divider()
st.caption(
    "Aplicación académica para CapitalGestión S.A.S. · Valores monetarios en millones de COP."
)
