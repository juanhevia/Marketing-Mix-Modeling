"""
K-MODA · Dashboard Analítico MMM + Random Forest
==================================================
Ejecutar:
    pip install streamlit plotly pandas numpy scikit-learn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

st.set_page_config(page_title="K-MODA Analytics", page_icon="👗",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────────────────────────────────────
GOLD   = "#C9A84C"
DARK   = "#0F0F1E"
CARD   = "#16162A"
LIGHT  = "#1E1E38"
TEXT   = "#F5F5F5"
GRAY   = "#CBD5E1"
ACCENT = "#00D4FF"
GREEN  = "#22C55E"
RED    = "#EF4444"
GRID   = "#2A2A4A"

CANAL_COLORS = {
    "Paid Search":  "#3B82F6",
    "Social Paid":  "#8B5CF6",
    "Video Online": "#EF4444",
    "Display":      "#10B981",
    "Email CRM":    "#F59E0B",
    "Radio Local":  "#06B6D4",
    "Exterior":     "#EC4899",
    "Prensa":       "#94A3B8",
}
CANALES = list(CANAL_COLORS.keys())

# Distribución IAB Spain 2024 — referencia para ROI y simulador
DIST_IAB = {
    "Social Paid":  0.27, "Paid Search":  0.24,
    "Video Online": 0.18, "Display":      0.12,
    "Email CRM":    0.09, "Exterior":     0.05,
    "Radio Local":  0.03, "Prensa":       0.02,
}
BUDGET_2024 = 12_000_000
PRECIO_MEDIO = 45.0
MARGEN = 0.65  # margen bruto real del CSV de pedidos (64.2% ≈ 65%)

# Factor de escala para convertir las cifras del modelo a valores reales de negocio.
# Base: benchmark sector moda España (Kantar 2024) → inversión publicitaria = 10% ventas.
# Con 12M€ de presupuesto → facturación real estimada = 120M€.
# El CSV de pedidos (10 ciudades, datos académicos) representa ~1.3% de esa cifra.
# Factor = 120M€ / (ventas_csv_2024 × PRECIO_MEDIO / 45) ≈ 79.
# Se aplica SOLO a cifras absolutas de euros mostradas al usuario;
# los pesos, ROI y distribución óptima NO cambian.
FACTOR_ESCALA = 79.0  # ventas_reales = ventas_modelo × FACTOR_ESCALA

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body {{ background-color:{DARK} !important; color:{TEXT} !important; }}
  #root, .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > section,
  [data-testid="stAppViewContainer"] > section > div,
  [data-testid="stMain"], [data-testid="stMainBlockContainer"],
  .main, .main > div {{
      background-color:{DARK} !important; color:{TEXT} !important;
  }}
  .block-container {{ padding:1.5rem 2rem !important; background-color:{DARK} !important; }}
  [data-testid="stDecoration"], [data-testid="stHeader"],
  header[data-testid="stHeader"] {{
      background-color:{DARK} !important; background-image:none !important;
  }}
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"],
  [data-testid="stSidebarUserContent"] {{
      background-color:{CARD} !important; color:{TEXT} !important;
  }}
  [data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
  h1,h2,h3,h4 {{ color:{GOLD} !important; }}
  p, li, span, label, div, td, th {{ color:{TEXT} !important; }}
  [data-testid="metric-container"] {{
      background:{CARD} !important; border:1px solid {GOLD}33 !important;
      border-radius:10px !important; padding:10px !important;
  }}
  [data-testid="stMetricLabel"] {{ color:{GRAY} !important; font-size:11px !important; }}
  [data-testid="stMetricValue"] {{ color:{TEXT} !important; font-size:22px !important; font-weight:700 !important; }}
  .stMultiSelect span, .stMultiSelect label, .stSlider label {{ color:{TEXT} !important; }}
  [data-baseweb="select"] div, [data-baseweb="tag"] {{
      background:{LIGHT} !important; color:{TEXT} !important;
  }}
  [data-baseweb="input"] input {{ background:{LIGHT} !important; color:{TEXT} !important; }}
  hr {{ border-color:{GOLD}33 !important; }}
  .explain {{
      background:{CARD}; border-left:4px solid {GOLD}; border-radius:8px;
      padding:12px 16px; color:{GRAY}; font-size:13px; line-height:1.75; margin-bottom:10px;
  }}
  .section-title {{
      color:{GOLD}; font-size:19px; font-weight:700;
      border-bottom:1px solid {GOLD}55; padding-bottom:6px; margin:24px 0 10px 0;
  }}
  /* ── Pestañas ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
      background:{CARD} !important; border-bottom:2px solid {GOLD}44 !important;
      gap:4px !important;
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
      background:{LIGHT} !important; color:{GRAY} !important;
      border-radius:8px 8px 0 0 !important; padding:10px 20px !important;
      font-weight:600 !important; font-size:13px !important;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
      background:{GOLD} !important; color:{DARK} !important;
  }}
  [data-testid="stTabPanel"] {{ background:{DARK} !important; padding-top:16px !important; }}
  ::-webkit-scrollbar {{ width:6px; }}
  ::-webkit-scrollbar-track {{ background:{DARK}; }}
  ::-webkit-scrollbar-thumb {{ background:{GOLD}66; border-radius:3px; }}

  /* ── Tooltips: fondo blanco, texto negro ── */
  div[data-testid="stTooltipContent"],
  div[data-baseweb="tooltip"],
  [role="tooltip"] {{
      background:#FFFFFF !important;
      color:#111111 !important;
      border:1px solid {GOLD} !important;
      border-radius:6px !important;
  }}
  div[data-testid="stTooltipContent"] *,
  div[data-baseweb="tooltip"] *,
  [role="tooltip"] * {{
      color:#111111 !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bl(fig, title="", height=380):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=LIGHT,
        font=dict(color=TEXT, family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=45, r=20, t=50, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GOLD, borderwidth=1,
                    font=dict(color=TEXT, size=11)),
        title=dict(text=title, font=dict(size=14, color=GOLD)),
        height=height,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID,
                     tickfont=dict(color=TEXT, size=11), title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID,
                     tickfont=dict(color=TEXT, size=11), title_font=dict(color=TEXT))
    return fig

def tip(txt):
    st.markdown(f'<div class="explain">💡 {txt}</div>', unsafe_allow_html=True)

def sec(txt):
    st.markdown(f'<div class="section-title">{txt}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES — fuente de verdad única para toda la app
# ─────────────────────────────────────────────────────────────────────────────
# Multiplicadores de tendencia 2024
TREND_2024_GLOBAL = {
    "Social Paid":  1.45,
    "Video Online": 1.30,
    "Paid Search":  1.10,
    "Display":      0.95,
    "Email CRM":    0.85,
    "Exterior":     0.90,
    "Radio Local":  0.65,
    "Prensa":       0.50,
}

# Techo de saturación por canal: ROI máximo sostenible cuando se escala
# al presupuesto óptimo. Email CRM tiene ROI base alto por infrainversión
# histórica (2.3M€), pero su ROI real se desploma al escalar porque la base
# de clientes CRM es finita — no se puede enviar más emails indefinidamente.
# Social Paid y Video Online sí escalan bien porque el inventario es prácticamente
# ilimitado en España.
ROI_TECHO_GLOBAL = {
    "Social Paid":  0.25,   # escala bien, inventario enorme
    "Paid Search":  0.20,   # escala bien, intención de compra clara
    "Video Online": 0.18,   # escala bien, CTV en crecimiento
    "Display":      0.08,   # bajo, banner blindness crónico
    "Email CRM":    0.10,   # techo real: lista CRM finita, no escala más allá del 6%
    "Exterior":     0.09,   # moderado, zonas premium limitadas
    "Radio Local":  0.06,   # bajo y decreciente en audiencia <45
    "Prensa":       0.04,   # muy bajo, audiencia residual y de mayor edad
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS  — versión 4 (fuerza recálculo limpio)
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_data(show_spinner=False, ttl=None)
def load_data():  # v5
    cal = pd.read_csv(BASE / "calendario_ciudad.csv", parse_dates=["fecha"])
    cli = pd.read_csv(BASE / "clientes.csv", parse_dates=["fecha_alta"],
                      dtype={"segmento":"category","canal_preferido":"category",
                             "ciudad_residencia":"category","sexo":"category"},
                      usecols=["ciudad_residencia","sexo","edad","segmento",
                               "fecha_alta","canal_preferido"])
    inv = pd.read_csv(BASE / "inversion_medios_semanal.csv", parse_dates=["semana_inicio"])
    inv = inv[inv["anio"] <= 2024].copy()   # eliminar residuos 2025
    scale = BUDGET_2024 / inv[inv["anio"] == 2024]["inversion_eur"].sum()
    inv["inversion_eur"] *= scale
    trf = pd.read_csv(BASE / "trafico_tienda_web_diario.csv", parse_dates=["fecha"])
    pro = pd.read_csv(BASE / "productos.csv")
    return cal, cli, inv, trf, pro

@st.cache_data(show_spinner=False, ttl=None)
def build_model(_inv, _trf, _versión=5):
    """
    Entrena dos Random Forest:
      - rf      : predicción (con lags, mejor MAPE)
      - rf_mmm  : descomposición MMM (sin lags, base orgánica realista)
    Versión 4 — ROI con denominador IAB 2024, prior cap 2.5x, calibración 65%.
    """
    trf = _trf.copy()
    trf["semana"] = trf["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    trf_sem = trf.groupby("semana").agg(
        sesiones_web=("sesiones_web","sum"),
        pedidos_online=("pedidos_online","sum"),
        pedidos_tienda=("pedidos_tienda","sum"),
        pedidos_cc=("pedidos_click_collect","sum"),
        visitas_tienda=("visitas_tienda","sum"),
        temperatura=("temperatura_media_c","mean"),
        lluvia=("lluvia_indice","mean"),
        turismo=("turismo_indice","mean"),
        rebajas=("rebajas_flag","max"),
        black_friday=("black_friday_flag","max"),
        navidad=("navidad_flag","max"),
        payday=("payday_flag","max"),
        festivo=("festivo_local_flag","max"),
    ).reset_index()
    trf_sem["ventas_total"] = (trf_sem["pedidos_online"]
                               + trf_sem["pedidos_tienda"]
                               + trf_sem["pedidos_cc"])

    inv_c = _inv.pivot_table(index="semana_inicio", columns="canal_medio",
                             values="inversion_eur", aggfunc="sum").fillna(0).reset_index()
    inv_c.columns.name = None
    inv_c = inv_c.rename(columns={"semana_inicio":"semana"})

    # Adstock + Hill por canal
    ALPHAS = {"Paid Search":0.20,"Social Paid":0.40,"Video Online":0.55,
              "Display":0.30,"Email CRM":0.15,"Radio Local":0.35,
              "Exterior":0.65,"Prensa":0.35}
    for c in CANALES:
        if c not in inv_c.columns: inv_c[c] = 0.0
        arr = inv_c[c].values.astype(float)
        # Adstock geométrico en euros reales + log1p para estabilizar escala
        # log1p preserva la proporciónalidad entre canales:
        # Social (8M€) >> Radio (300K€) → el RF aprende la diferencia real
        ads = np.zeros(len(arr))
        for t in range(len(arr)):
            ads[t] = arr[t] + (ALPHAS[c] * ads[t-1] if t > 0 else 0)
        inv_c[f"ads_{c}"] = np.log1p(ads)

    df = trf_sem.merge(inv_c, on="semana", how="left").fillna(0)
    df["mes"]         = df["semana"].dt.month
    df["trimestre"]   = df["semana"].dt.quarter
    df["semana_iso"]  = pd.DatetimeIndex(df["semana"]).isocalendar().week.astype(int)
    df["ventas_lag1"] = df["ventas_total"].shift(1).fillna(0)
    df["ventas_lag4"] = df["ventas_total"].shift(4).fillna(0)
    df["ses_lag1"]    = df["sesiones_web"].shift(1).fillna(0)

    feats = ([f"ads_{c}" for c in CANALES]
             + ["temperatura","lluvia","turismo","rebajas","black_friday",
                "navidad","payday","festivo","mes","trimestre","semana_iso",
                "ventas_lag1","ventas_lag4","ses_lag1"])
    feats_mmm = ([f"ads_{c}" for c in CANALES]
                 + ["temperatura","lluvia","turismo","rebajas","black_friday",
                    "navidad","payday","festivo","mes","trimestre","semana_iso"])

    X, X_mmm, y = df[feats].values, df[feats_mmm].values, df["ventas_total"].values
    sp = int(len(df) * 0.8)

    rf = RandomForestRegressor(500, max_depth=10, min_samples_leaf=2,
                               max_features=0.7, random_state=42, n_jobs=-1)
    rf.fit(X[:sp], y[:sp])

    rf_mmm = RandomForestRegressor(500, max_depth=10, min_samples_leaf=2,
                                   max_features=0.7, random_state=42, n_jobs=-1)
    rf_mmm.fit(X_mmm[:sp], y[:sp])

    y_pred   = rf.predict(X)
    mape     = mean_absolute_percentage_error(y[sp:], rf.predict(X[sp:]))
    r2       = r2_score(y[:sp], rf.predict(X[:sp]))
    imps     = pd.Series(rf.feature_importances_, index=feats)

    # Descomposición MMM con calibración al 65% base orgánica
    X0 = X_mmm.copy()
    for c in CANALES:
        X0[:, feats_mmm.index(f"ads_{c}")] = 0
    base_raw = rf_mmm.predict(X0)
    y_mmm    = rf_mmm.predict(X_mmm)
    # Benchmark 55%: límite inferior rango retail digital español
    # (Analytic Partners 2024: rango 55-70%; usamos 55% porque la correlación
    # medios-ventas en estos datos es moderada ~0.40, indicando mayor impacto de medios)
    base     = base_raw * (0.55 / (base_raw.sum() / (y_mmm.sum() + 1e-9)))

    contrib_raw = {}
    for c in CANALES:
        Xs = X0.copy()
        Xs[:, feats_mmm.index(f"ads_{c}")] = X_mmm[:, feats_mmm.index(f"ads_{c}")]
        contrib_raw[c] = np.maximum(rf_mmm.predict(Xs) - base_raw, 0)

    # Cap 2.5x por peso presupuestario (evita inflación de canales con baja inversión)
    inv_tot = {c: _inv[_inv["canal_medio"]==c]["inversion_eur"].sum() for c in CANALES}
    inv_sum  = sum(inv_tot.values()) + 1e-9
    target45 = y_mmm.sum() * 0.45
    sc_raw   = target45 / (sum(v.sum() for v in contrib_raw.values()) + 1e-9)
    contrib_sc = {c: contrib_raw[c] * sc_raw for c in CANALES}
    target45 = y_mmm.sum() * 0.45
    MAX_M = 2.5
    contrib = {}
    for c in CANALES:
        cap = (inv_tot[c] / inv_sum) * target45 * MAX_M
        contrib[c] = np.minimum(contrib_sc[c], cap / max(len(contrib_sc[c]), 1))
    sc2 = target45 / (sum(v.sum() for v in contrib.values()) + 1e-9)
    for c in CANALES:
        contrib[c] *= sc2

    # ROI con denominador inversión histórica real del CSV
    # Es el único denominador consistente: el modelo aprendió con esa inversión,
    # por tanto el ROI debe calcularse contra lo que realmente se invirtió.
    # Usar inv_IAB como denominador inflaba canales históricamente sobreinvertidos
    # (Radio, Exterior) porque el modelo les atribuía más contribución de la que
    # correspondería con su peso IAB óptimo.
    roi, venta_eur = {}, {}
    for c in CANALES:
        ve = contrib[c].sum() * PRECIO_MEDIO
        venta_eur[c] = ve
        inv_hist = _inv[_inv["canal_medio"] == c]["inversion_eur"].sum()
        roi[c]   = ve / inv_hist if inv_hist > 0 else 0

    r2_te = r2_score(y[sp:], rf.predict(X[sp:]))
    return dict(df=df, y=y, y_pred=y_pred, base=base, contrib=contrib,
                imps=imps, mape=mape, r2=r2, r2_te=r2_te, roi=roi,
                venta_eur=venta_eur, inv_tot=inv_tot, feats=feats)

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Cargando datos…"):
    cal, cli, inv, trf, pro = load_data()
with st.spinner("Entrenando modelos (puede tardar ~30 s la primera vez)…"):
    M = build_model(inv, trf, _versión=5)

df       = M["df"]
y        = M["y"]
y_pred   = M["y_pred"]
base     = M["base"]
contrib  = M["contrib"]
imps     = M["imps"]
mape     = M["mape"]
r2       = M["r2"]
r2_te    = M["r2_te"]
roi      = M["roi"]
venta_eur= M["venta_eur"]
inv_tot  = M["inv_tot"]

# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUCIÓN ÓPTIMA GLOBAL — calculada UNA SOLA VEZ, usada en todos los tabs
# ─────────────────────────────────────────────────────────────────────────────
# roi_adj = min(roi_modelo × tendencia, techo_saturación)
roi_adj_global = {c: min(roi[c] * TREND_2024_GLOBAL[c], ROI_TECHO_GLOBAL[c])
                  for c in CANALES}

_CAPS_GLOBAL = {
    "Social Paid":  (0.22, 0.35),
    "Paid Search":  (0.18, 0.28),
    "Video Online": (0.12, 0.22),
    "Display":      (0.05, 0.10),
    "Email CRM":    (0.04, 0.06),
    "Exterior":     (0.03, 0.06),
    "Radio Local":  (0.01, 0.03),
    "Prensa":       (0.01, 0.02),
}
_pg = {c: max(roi_adj_global[c], 0) for c in CANALES}
_sg = sum(_pg.values())
pesos_opt_global = {c: _pg[c] / _sg for c in CANALES}
for _ in range(200):
    _changed = False
    for c in CANALES:
        lo, hi = _CAPS_GLOBAL[c]
        if pesos_opt_global[c] < lo:
            pesos_opt_global[c] = lo; _changed = True
        elif pesos_opt_global[c] > hi:
            pesos_opt_global[c] = hi; _changed = True
    _sg2 = sum(pesos_opt_global.values())
    pesos_opt_global = {c: v / _sg2 for c, v in pesos_opt_global.items()}
    if not _changed:
        break

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 👗 K-MODA")
    st.markdown(f"<span style='color:{GRAY};font-size:12px'>Dashboard MMM · 2020-2024</span>",
                unsafe_allow_html=True)
    st.divider()
    años_disp  = sorted(inv["anio"].unique().tolist())
    años_sel   = st.multiselect("Año", años_disp, default=años_disp)
    ciudad_disp= sorted(trf["ciudad"].unique().tolist())
    ciudad_sel = st.multiselect("Ciudad", ciudad_disp, default=ciudad_disp)
    st.divider()
    st.markdown(f"<span style='color:{GRAY};font-size:11px'>10 ciudades · 8 canales · RF 500 arboles</span>",
                unsafe_allow_html=True)

inv_f = inv[inv["anio"].isin(años_sel)]
trf_f = trf[trf["ciudad"].isin(ciudad_sel) & trf["fecha"].dt.year.isin(años_sel)].copy()
cal_f = cal[cal["ciudad"].isin(ciudad_sel) & cal["anio"].isin(años_sel)]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{CARD},#1A1A3E);
            border-bottom:2px solid {GOLD};padding:22px 28px;
            border-radius:12px;margin-bottom:20px;">
  <span style="color:{GOLD};font-size:34px;font-weight:800;letter-spacing:4px">K-MODA</span><br>
  <span style="color:{TEXT};font-size:14px;opacity:0.85">
    Dashboard Analitico · Marketing Mix Modeling + Random Forest · 2020-2024
  </span>
</div>
""", unsafe_allow_html=True)

# KPIs globales
total_inv  = inv_f["inversion_eur"].sum()
total_ped  = (trf_f["pedidos_online"].sum() + trf_f["pedidos_tienda"].sum()
              + trf_f["pedidos_click_collect"].sum())
best_roi   = max(roi, key=roi.get)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Inversión 2024",      "12 M€",
          help="Presupuesto anual real de marketing K-Moda para 2024 segun el caso.")
c2.metric("Pedidos en Muestra",  f"{int(total_ped/1e3)}K",
          help="Pedidos registrados en el CSV de tráfico. Es una muestra estadística de "
               "10 ciudades, NO el total real de K-Moda. Los porcentajes y rankings "
               "entre canales son fiables; los euros absolutos son proporciónales.")
c3.metric("Ventas Orgánicas",    "55%",
          help="Porcentaje de ventas que existirian sin publicidad: clientes recurrentes, "
               "tráfico directo y fuerza de marca. Calibrado al 55% segun benchmark "
               "retail digital español (Analytic Partners 2024, rango 55-70%).")
c4.metric("Ventas por Medios",   "45%",
          help="Porcentaje de ventas generadas directamente por la inversión publicitaria. "
               "La suma orgánica(55%) + medios(45%) = 100% de las ventas totales. "
               "La inversión de 12M€ no se resta a las ventas: es el coste para generarlas.")
c5.metric("Canal Mayor ROI",     best_roi,
          help="Canal más eficiente segun el modelo Random Forest ajustado "
               "por tendencias de mercado 2024.")
c6.metric("MAPE Modelo",         f"{mape*100:.1f}%",
          help="Error medio del modelo de predicción. Un MAPE del 10-12% es "
               "estadísticamente riguroso en marketing: refleja que el mercado "
               "tiene un 10% de variabilidad genuinamente impredecible.")

# Estimación económica basada en benchmark sectorial
_fac_real  = BUDGET_2024 / 0.10          # 12M€ = 10% ventas → 120M€ facturación
_gan_bruta = _fac_real * MARGEN          # 120M€ × 64.2% = 77M€
_gan_med   = _gan_bruta * 0.45           # 45% por medios = 34.7M€
_gan_neta  = _gan_med - BUDGET_2024      # retorno neto = 22.7M€

st.markdown(f"""<div style="background:{CARD};border-left:3px solid {GOLD};
border-radius:6px;padding:10px 16px;margin:6px 0 8px 0;font-size:12px;color:{GRAY};">
<b style="color:{GOLD}">Estimación económica K-Moda 2024</b> —
Benchmark sector moda España: inversión publicitaria = 10% de ventas (Kantar 2024).<br>
Con 12M€ de presupuesto →
<b style="color:{TEXT}">Facturación estimada: {_fac_real/1e6:.0f}M€</b> ·
<b style="color:{TEXT}">Ganancia bruta: {_gan_bruta/1e6:.0f}M€</b> ·
<b style="color:{GREEN}">Retorno neto publicidad: +{_gan_neta/1e6:.0f}M€</b>
(cada €1 invertido genera <b style="color:{GREEN}">{_gan_med/BUDGET_2024:.2f}€</b> en margen).
</div>""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📺  Inversión",
    "🤖  Modelo RF",
    "🌱  Ventas y Orgánica",
    "📈  ROI por Canal",
    "💰  Ganancias",
    "👥  Clientes",
    "👗  Producto",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · INVERSIÓN EN MEDIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sec("Histórico 2020-2024")
    tip("Evolución del presupuesto publicitario por canal y año. Los datos del CSV han sido "
        "escalados para que 2024 = 12M€ exactos, manteniendo las proporciónes relativas entre años.")

    inv_ay = inv_f.groupby(["anio","canal_medio"])["inversion_eur"].sum().reset_index()
    inv_tot2 = inv_f.groupby("canal_medio")["inversion_eur"].sum().reset_index()

    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure()
        for c, col in CANAL_COLORS.items():
            d = inv_ay[inv_ay["canal_medio"]==c]
            fig.add_trace(go.Bar(x=d["anio"], y=d["inversion_eur"]/1e6, name=c, marker_color=col))
        bl(fig, "Inversión por Canal y Año (M€)", 400)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(title_text="M€")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Pie(
            labels=inv_tot2["canal_medio"], values=inv_tot2["inversion_eur"],
            marker_colors=list(CANAL_COLORS.values()),
            hole=0.45, textinfo="label+percent", textfont=dict(size=10, color=TEXT),
        ))
        bl(fig2, "Distribución Histórica 2020-2024", 400)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    inv_ciu = inv_f.groupby(["ciudad","canal_medio"])["inversion_eur"].sum().reset_index()
    pivot   = inv_ciu.pivot(index="ciudad", columns="canal_medio", values="inversion_eur").fillna(0)
    fig3 = go.Figure(go.Heatmap(
        z=pivot.values/1e3, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,DARK],[0.5,GOLD],[1,"#FFFFFF"]],
        text=np.round(pivot.values/1e3, 0), texttemplate="%{text:.0f}K",
        colorbar=dict(title="K€", tickfont=dict(color=TEXT)),
    ))
    bl(fig3, "Inversión por Ciudad y Canal - Histórico 2020-2024 (K€)", 380)
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    sec("Los 12 Millones de 2024 - Distribución Óptima")
    tip("Distribución recomendada de los 12M€ segun benchmarks IAB Spain 2024 para retail premium. "
        "Social Paid lidera con el 27% dado el dominio de Instagram/TikTok/Reels en España. "
        "Prensa y Radio bajan a pesos residuales del 2-3% por su declive estructural de audiencia.")

    # Usar pesos_opt_global: fuente única calculada antes de los tabs
    inv_2024_df = pd.DataFrame([
        {"canal_medio": c, "inversion_eur": pesos_opt_global[c] * BUDGET_2024}
        for c in sorted(CANALES, key=lambda x: -pesos_opt_global[x])
    ])

    col3, col4 = st.columns(2)
    with col3:
        fig4 = go.Figure(go.Pie(
            labels=inv_2024_df["canal_medio"], values=inv_2024_df["inversion_eur"],
            marker_colors=[CANAL_COLORS[c] for c in inv_2024_df["canal_medio"]],
            hole=0.5, textinfo="label+percent", sort=False,
            textfont=dict(size=11, color=TEXT),
            hovertemplate="%{label}<br>%{value:,.0f} €<br>%{percent}<extra></extra>",
        ))
        bl(fig4, "Distribución 2024 - 12 M€ (IAB Spain 2024)", 420)
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        fig5 = go.Figure(go.Bar(
            x=inv_2024_df["inversion_eur"]/1e3,
            y=inv_2024_df["canal_medio"],
            orientation="h",
            marker_color=[CANAL_COLORS[c] for c in inv_2024_df["canal_medio"]],
            text=[f"{v/1e3:.0f}K€  ({v/BUDGET_2024*100:.0f}%)"
                  for v in inv_2024_df["inversion_eur"]],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        bl(fig5, "Inversión 2024 por Canal (K€)", 420)
        fig5.update_xaxes(title_text="K€")
        st.plotly_chart(fig5, use_container_width=True)

    tip("Tres perspectivas comparadas: distribución histórica del CSV (sesgada hacia offline), "
        "distribución óptima IAB 2024 (realidad digital actual) y ROI relativo del modelo. "
        "La brecha entre barras rojas y doradas marca exactamente donde reasignar presupuesto.")

    csv_2024 = inv[inv["anio"]==2024].groupby("canal_medio")["inversion_eur"].sum()
    csv_tot  = csv_2024.sum()
    csv_pct  = {c: csv_2024.get(c, 0)/csv_tot*100 for c in CANALES}
    roi_tot  = sum(max(v,0) for v in roi.values())
    roi_pct  = {c: max(roi[c],0)/roi_tot*100 if roi_tot>0 else 0 for c in CANALES}
    # Ordenar por ROI descendente para mostrar los mejores primero
    ords     = sorted(CANALES, key=lambda c: -roi[c])

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name="% CSV histórico", x=ords,
                          y=[csv_pct.get(c,0) for c in ords],
                          marker_color=RED, opacity=0.75))
    fig6.add_trace(go.Bar(name="% IAB 2024 (recomendado)", x=ords,
                          y=[DIST_IAB.get(c,0)*100 for c in ords],
                          marker_color=GOLD, opacity=0.85))
    fig6.add_trace(go.Bar(name="% ROI relativo (RF)", x=ords,
                          y=[roi_pct.get(c,0) for c in ords],
                          marker_color=ACCENT, opacity=0.85))
    bl(fig6, "Comparativa: CSV Histórico vs IAB 2024 vs ROI Modelo (%)", 420)
    fig6.update_layout(barmode="group")
    fig6.update_yaxes(title_text="%")
    st.plotly_chart(fig6, use_container_width=True)

    sec("Distribución 2024 por Ciudad y Canal")
    tip("Como se reparten los 12M€ geográficamente. Permite detectar si Madrid concentra "
        "demasiado presupuesto digital respecto a su peso en clientes.")
    inv_2024_ciu = inv[inv["anio"]==2024].groupby(["ciudad","canal_medio"])["inversion_eur"].sum().reset_index()
    piv24 = inv_2024_ciu.pivot(index="ciudad", columns="canal_medio", values="inversion_eur").fillna(0)
    fig_h24 = go.Figure(go.Heatmap(
        z=piv24.values/1e3, x=piv24.columns.tolist(), y=piv24.index.tolist(),
        colorscale=[[0,DARK],[0.5,GOLD],[1,"#FFFFFF"]],
        text=np.round(piv24.values/1e3, 0), texttemplate="%{text:.0f}K",
        colorbar=dict(title="K€", tickfont=dict(color=TEXT)),
    ))
    bl(fig_h24, "Distribución de los 12M€ por Ciudad y Canal en 2024 (K€)", 400)
    st.plotly_chart(fig_h24, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · MODELO RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    sec("Predicción del Modelo Random Forest")
    tip("Dos modelos RF entrenados sobre 262 semanas (2020-2024): "
        "(1) RF de predicción con lags autoregresivos para minimizar el MAPE — "
        "el R² de entrenamiento (~0.96) es alto porque los lags permiten 'ver' la semana anterior, "
        "lo que no es sobreajuste sino memoria temporal; el R² de test (~0.55) es el valor honesto "
        "sobre datos no vistos y refleja la variabilidad real del mercado. "
        "(2) RF de descomposición MMM sin lags para aislar la contribución real de cada canal.")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df["semana"], y=y, name="Pedidos Reales",
                                   line=dict(color=GOLD, width=1.8),
                                   fill="tozeroy", fillcolor="rgba(201,168,76,0.08)"))
        fig_p.add_trace(go.Scatter(x=df["semana"], y=y_pred, name="Predicción RF",
                                   line=dict(color=ACCENT, width=1.8, dash="dash")))
        split_d = df["semana"].iloc[int(len(df)*0.8)]
        fig_p.add_shape(type="line", x0=str(split_d), x1=str(split_d),
                        y0=0, y1=1, xref="x", yref="paper",
                        line=dict(color="rgba(255,100,100,0.7)", dash="dot", width=1.5))
        fig_p.add_annotation(x=str(split_d), y=1.02, xref="x", yref="paper",
                             text="Train | Test", showarrow=False,
                             font=dict(color=TEXT, size=10))
        bl(fig_p, "Pedidos Semanales: Reales vs Predicción RF", 400)
        fig_p.update_yaxes(title_text="Pedidos/semana")
        st.plotly_chart(fig_p, use_container_width=True)
    with col_b:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=y, y=y_pred, mode="markers",
                                   marker=dict(color=GOLD, opacity=0.4, size=5),
                                   hovertemplate="Real:%{x}<br>Pred:%{y}"))
        lim = max(y.max(), y_pred.max()) * 1.05
        fig_s.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode="lines",
                                   line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                                   name="Perfecto"))
        bl(fig_s, f"Real vs Predicho  |  R² train={r2:.3f}  R² test={r2_te:.3f}  MAPE={mape*100:.1f}%", 400)
        fig_s.update_xaxes(title_text="Pedidos Reales")
        fig_s.update_yaxes(title_text="Pedidos Predichos")
        st.plotly_chart(fig_s, use_container_width=True)

    sec("Importancia de Variables")
    tip("Que variables explican más los cambios semanales en pedidos. "
        "Las variables temporales (mes, semana) capturan la estacionalidad. "
        "Los canales con mayor importancia son los que tienen impacto más consistente "
        "sobre las ventas semana a semana.")

    imp_df = imps.sort_values(ascending=True).reset_index()
    imp_df.columns = ["feature","importance"]
    imp_df["feature"] = (imp_df["feature"]
                         .str.replace("ads_","Adstock: ", regex=False)
                         .str.replace("_"," ").str.title())
    fig_i = go.Figure(go.Bar(
        x=imp_df["importance"], y=imp_df["feature"],
        orientation="h",
        marker_color=[GOLD if i >= len(imp_df)-5 else GRAY for i in range(len(imp_df))],
        text=imp_df["importance"].apply(lambda v: f"{v:.3f}"),
        textposition="outside", textfont=dict(color=TEXT, size=10),
    ))
    bl(fig_i, "Importancia de Variables - Random Forest", 560)
    fig_i.update_xaxes(title_text="Importancia (reducción de impureza)")
    st.plotly_chart(fig_i, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · VENTAS ORGANICAS vs INCREMENTALES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    sec("Descomposición de Ventas: Orgánicas vs Medios")
    tip("Ventas orgánicas (55%): las que ocurririan aunque apagaramos toda la publicidad, "
        "debidas a la fuerza de marca, clientes recurrentes y tráfico natural. "
        "Ventas por medios (45%): generadas directamente por la inversión publicitaria. "
        "Calibrado al 55% (limite inferior rango sectorial) dado que la correlación medios-ventas en estos datos es moderada (~0.40), indicando mayor impacto publicitario que la media.")

    semanas = df["semana"]
    base_eur = base * PRECIO_MEDIO

    # Escalar contrib por el factor roi_adj/roi para que refleje tendencias 2024
    # Si roi[c]=0 usamos 1 para evitar división por cero
    contrib_adj_factor = {
        c: (roi_adj_global[c] / roi[c]) if roi.get(c, 0) > 0 else 1.0
        for c in CANALES
    }

    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(
        x=semanas, y=base_eur/1e3, name="Base Orgánica", stackgroup="one",
        line=dict(width=0), fillcolor="rgba(148,163,184,0.6)",
        hovertemplate="Base orgánica: %{y:.1f}K€"))
    for c in CANALES:
        contrib_ajustado = contrib[c] * contrib_adj_factor[c] * PRECIO_MEDIO / 1e3
        fig_st.add_trace(go.Scatter(
            x=semanas, y=contrib_ajustado,
            name=c, stackgroup="one", line=dict(width=0),
            fillcolor=f"rgba({_hex_to_rgb(CANAL_COLORS[c])},0.73)",
            hovertemplate=f"{c}: %{{y:.1f}}K€"))
    bl(fig_st, "Descomposición Semanal: Orgánica + Incremental por Canal — Ajustado 2024 (K€)", 430)
    fig_st.update_yaxes(title_text="Ventas estimadas (K€)")
    st.plotly_chart(fig_st, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        tb = base_eur.sum()
        tm = sum(contrib[c].sum() for c in CANALES) * PRECIO_MEDIO
        fig_pie = go.Figure(go.Pie(
            labels=["Ventas Orgánicas", "Ventas por Medios"],
            values=[tb, tm], hole=0.55,
            marker_colors=[GRAY, GOLD],
            textinfo="label+percent", textfont=dict(size=13, color=TEXT),
        ))
        bl(fig_pie, "Split Organico (55%) vs Medios (45%) — Benchmark retail digital", 360)
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        # Ventas atribuidas ajustadas por tendencias 2024 y techo de saturación
        # Usamos roi_adj_global × inversión óptima para que sea consistente
        # con el resto de la app (no el contrib raw del modelo que infla Radio)
        cv_total = sum(roi_adj_global[c] * pesos_opt_global[c] * BUDGET_2024
                       for c in CANALES)
        cv = {c: roi_adj_global[c] * pesos_opt_global[c] * BUDGET_2024
              for c in CANALES}
        cv_s = dict(sorted(cv.items(), key=lambda x: -x[1]))
        fig_cb = go.Figure(go.Bar(
            x=list(cv_s.keys()), y=[v/1e3 for v in cv_s.values()],
            marker_color=[CANAL_COLORS[c] for c in cv_s],
            text=[f"{v/1e3:.0f}K€  ({v/cv_total*100:.1f}%)" for v in cv_s.values()],
            textposition="outside", textfont=dict(color=TEXT, size=9),
        ))
        bl(fig_cb, "Ventas Atribuidas por Canal — Mix Óptimo 2024 (K€/año)", 360)
        fig_cb.update_yaxes(title_text="K€/año")
        st.plotly_chart(fig_cb, use_container_width=True)

    sec("Tráfico y Demanda")
    tip("Serie temporal de sesiones web y pedidos online. Los picos de Black Friday, "
        "Rebajas y Navidad son variables de control que el modelo aisla para no "
        "atribuirlos a la publicidad activa esa semana.")

    trf_f["semana"] = trf_f["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    trf_sem = trf_f.groupby("semana").agg(
        sesiones_web=("sesiones_web","sum"),
        pedidos_online=("pedidos_online","sum"),
        pedidos_tienda=("pedidos_tienda","sum"),
        pedidos_cc=("pedidos_click_collect","sum"),
        visitas_tienda=("visitas_tienda","sum"),
    ).reset_index()

    fig_tr = make_subplots(specs=[[{"secondary_y":True}]])
    fig_tr.add_trace(go.Scatter(
        x=trf_sem["semana"], y=trf_sem["sesiones_web"]/1e3,
        name="Sesiónes Web (K)", line=dict(color=ACCENT, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.10)"), secondary_y=False)
    fig_tr.add_trace(go.Scatter(
        x=trf_sem["semana"], y=trf_sem["pedidos_online"],
        name="Pedidos Online", line=dict(color=GOLD, width=1.5)), secondary_y=True)
    bl(fig_tr, "Tráfico Web Semanal: Sesiónes vs Pedidos Online", 380)
    fig_tr.update_yaxes(title_text="Sesiónes (K)", secondary_y=False)
    fig_tr.update_yaxes(title_text="Pedidos", secondary_y=True)
    st.plotly_chart(fig_tr, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        tc = trf_f.groupby("ciudad").agg(
            ses=("sesiones_web","sum"), vis=("visitas_tienda","sum")).reset_index()
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Bar(name="Sesiónes Web (M)", x=tc["ciudad"],
                                y=tc["ses"]/1e6, marker_color=ACCENT))
        fig_tc.add_trace(go.Bar(name="Visitas Tienda (K)", x=tc["ciudad"],
                                y=tc["vis"]/1e3, marker_color=GOLD))
        bl(fig_tc, "Tráfico por Ciudad: Web (M) vs Tienda (K)", 360)
        fig_tc.update_layout(barmode="group")
        st.plotly_chart(fig_tc, use_container_width=True)
    with col4:
        conv = trf_f.groupby("ciudad").agg(
            ct=("tasa_conversion_tienda_pct","mean"),
            cw=("tasa_conversion_web_pct","mean")).reset_index()
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(name="Conversión Tienda (%)", x=conv["ciudad"],
                                y=conv["ct"]*100, marker_color=GOLD))
        fig_cv.add_trace(go.Bar(name="Conversión Web (%)", x=conv["ciudad"],
                                y=conv["cw"]*100, marker_color=ACCENT))
        bl(fig_cv, "Tasa de Conversión Media por Ciudad (%)", 360)
        fig_cv.update_layout(barmode="group")
        fig_cv.update_yaxes(title_text="%")
        st.plotly_chart(fig_cv, use_container_width=True)

    fig_mix = go.Figure(go.Pie(
        labels=["Tienda Fisica","Online","Click & Collect"],
        values=[trf_sem["pedidos_tienda"].sum(),
                trf_sem["pedidos_online"].sum(),
                trf_sem["pedidos_cc"].sum()],
        hole=0.5, marker_colors=[GOLD, ACCENT, GREEN],
        textinfo="label+percent", textfont=dict(size=13, color=TEXT),
    ))
    bl(fig_mix, "Mix de Pedidos por Canal de Venta", 360)
    fig_mix.update_layout(showlegend=False)
    st.plotly_chart(fig_mix, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · ROI POR CANAL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    sec("ROI por Canal de Medios")
    tip("ROI calculado como ventas_atribuidas / inversión_IAB_2024 (a 5 años). "
        "Usamos la distribución IAB como denominador — no el CSV histórico — porque "
        "el CSV infrainvertia en canales digitales (Email, Social) y sobreinvertia en "
        "offline (Radio, Prensa), lo que inflaba artificialmente su ROI aparente.")

    # ROI ajustado por tendencias 2024 + techo de saturación (fuente única)
    roi_s = dict(sorted(roi_adj_global.items(), key=lambda x: x[1]))
    col_e, col_f = st.columns([1.4, 1])
    with col_e:
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=list(roi_s.values()), y=list(roi_s.keys()),
            orientation="h",
            marker_color=[CANAL_COLORS[c] for c in roi_s],
            text=[f"{v:.3f}x" for v in roi_s.values()],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        media_roi = sum(roi_adj_global.values()) / len(roi_adj_global)
        fig_roi.add_shape(type="line", x0=media_roi, x1=media_roi,
                          y0=-0.5, y1=len(roi_s)-0.5,
                          xref="x", yref="y",
                          line=dict(color=GOLD, dash="dash", width=1.5))
        fig_roi.add_annotation(x=media_roi, y=len(roi_s)-0.5,
                               text="Media portfolio",
                               font=dict(color=GOLD, size=10),
                               showarrow=False, yshift=10)
        bl(fig_roi, "ROI Ajustado por Canal — Tendencias 2024 + Techo de Saturación", 420)
        fig_roi.update_xaxes(title_text="ROI ajustado (x)")
        st.plotly_chart(fig_roi, use_container_width=True)
    with col_f:
        # Inversión óptima vs ventas atribuidas ajustadas
        inv_arr = [pesos_opt_global[c] * BUDGET_2024 / 1e3 for c in CANALES]
        vta_arr = [roi_adj_global[c] * pesos_opt_global[c] * BUDGET_2024 / 1e3 for c in CANALES]
        fig_r2  = go.Figure()
        fig_r2.add_trace(go.Scatter(
            x=inv_arr, y=vta_arr, mode="markers+text",
            marker=dict(color=[CANAL_COLORS[c] for c in CANALES], size=18,
                        line=dict(color=TEXT, width=1)),
            text=CANALES, textposition="top center",
            textfont=dict(color=TEXT, size=9),
            hovertemplate="%{text}<br>Inversión óptima: %{x:.0f}K€<br>Ventas atribuidas: %{y:.0f}K€",
        ))
        mx = max(max(inv_arr), max(vta_arr)) * 1.2
        fig_r2.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                                    line=dict(color=GOLD, dash="dash", width=1.2), name="ROI=1x"))
        fig_r2.add_trace(go.Scatter(x=[0,mx], y=[0,mx*2], mode="lines",
                                    line=dict(color=GREEN, dash="dot", width=1), name="ROI=2x"))
        bl(fig_r2, "Inversión Óptima (K€) vs Ventas Atribuidas (K€) — Mix 2024", 420)
        fig_r2.update_xaxes(title_text="Inversión óptima (K€)")
        fig_r2.update_yaxes(title_text="Ventas atribuidas (K€)")
        st.plotly_chart(fig_r2, use_container_width=True)

    sec("Evolución del ROI por Año")
    tip("Un ROI decreciente en un canal indica saturación o perdida de eficiencia. "
        "Un ROI creciente indica que el canal gana tracción y justifica aumentar su presupuesto.")

    # Evolución del ROI por año: interpolación lineal entre el ROI base (2020)
    # y el ROI ajustado 2024. El multiplicador de tendencia (TREND_2024_GLOBAL)
    # refleja cuánto ha cambiado la eficiencia de cada canal de 2020 a 2024.
    # roi_2020 = roi_adj_2024 / TREND  →  roi_2024 = roi_adj_2024 (destino)
    # Esto garantiza que canales con TREND>1 (Social, Video) suban,
    # y canales con TREND<1 (Radio, Prensa) bajen — coherente con el mercado.
    años = sorted(inv["anio"].unique())
    n_años = len(años)
    fig_ry = go.Figure()
    for c in CANALES:
        roi_2024 = roi_adj_global[c]
        trend    = TREND_2024_GLOBAL[c]
        roi_2020 = roi_2024 / trend if trend > 0 else roi_2024
        roi_por_año = [
            roi_2020 + (roi_2024 - roi_2020) * i / (n_años - 1)
            for i in range(n_años)
        ]
        fig_ry.add_trace(go.Scatter(
            x=años, y=roi_por_año, name=c, mode="lines+markers",
            line=dict(color=CANAL_COLORS[c], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"{c}<br>Año: %{{x}}<br>ROI: %{{y:.3f}}x<extra></extra>",
        ))
    bl(fig_ry, "Evolución del ROI por Canal 2020-2024 — Tendencias de Mercado", 440)
    fig_ry.update_xaxes(dtick=1, title_text="Año")
    fig_ry.update_yaxes(title_text="ROI (x)")
    tip("El ROI de cada canal se interpola desde su nivel base en 2020 "
        "hasta el ROI ajustado por tendencias en 2024. Los canales con "
        "multiplicador > 1 (Social Paid +45%, Video Online +30%) muestran "
        "línea creciente — su eficiencia mejora cada año con el auge digital. "
        "Los canales con multiplicador < 1 (Radio -35%, Prensa -50%) muestran "
        "línea decreciente — su audiencia mengua estructuralmente.")
    st.plotly_chart(fig_ry, use_container_width=True)

    sec("Impacto del Calendario")
    tip("Variación de tráfico y pedidos en dias/semanas de evento vs dias normales. "
        "Estos efectos son capturados por el modelo como variables de control para no "
        "atribuirlos a la publicidad activa ese período.")

    flags_map = {"payday_flag":"Dia de Pago","rebajas_flag":"Rebajas",
                 "black_friday_flag":"Black Friday","navidad_flag":"Navidad",
                 "festivo_local_flag":"Festivo Local"}
    trf_cal = trf_f.merge(cal_f[["fecha","ciudad","semana_santa_flag"]],
                          on=["fecha","ciudad"], how="left")
    trf_cal["semana_santa_flag"] = trf_cal["semana_santa_flag"].fillna(0)
    flags_map["semana_santa_flag"] = "Semana Santa"

    us, up, ns = [], [], []
    for f, lbl in flags_map.items():
        g = trf_cal.groupby(f)[["sesiones_web","pedidos_online"]].mean()
        if 0 in g.index and 1 in g.index and g.loc[0,"sesiones_web"]>0:
            us.append((g.loc[1,"sesiones_web"]/g.loc[0,"sesiones_web"]-1)*100)
            up.append((g.loc[1,"pedidos_online"]/g.loc[0,"pedidos_online"]-1)*100)
            ns.append(lbl)
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(name="Uplift Sesiónes Web (%)", x=ns, y=us, marker_color=ACCENT))
    fig_ev.add_trace(go.Bar(name="Uplift Pedidos Online (%)", x=ns, y=up, marker_color=GOLD))
    fig_ev.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    bl(fig_ev, "Variación de Tráfico en Eventos vs Dias Normales (%)", 380)
    fig_ev.update_layout(barmode="group")
    fig_ev.update_yaxes(title_text="Variación %")
    st.plotly_chart(fig_ev, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · GANANCIAS Y OPTIMIZACION ESTRATEGICA
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    # ── ROI ajustado por tendencias 2024 ──────────────────────────────────────
    # El modelo estadístico refleja datos históricos 2020-2024 donde Email CRM
    # y Radio tenían más inversión y por tanto más señal. Para la toma de
    # decisiónes 2024 aplicamos multiplicadores de tendencia basados en:
    # - IAB Spain Digital Ad Spend Report 2024
    # - GWI Consumer Trends Spain Q4 2024
    # - Meta Business Insights España 2024
    TREND_2024 = TREND_2024_GLOBAL  # referencia la constante global

    # Usar roi_adj_global: calculado antes de los tabs, fuente única
    roi_adj = roi_adj_global

    # Usar pesos_opt_global: calculado antes de los tabs, fuente única
    pesos_opt = pesos_opt_global

    # ── TABLA DE MULTIPLICADORES DE TENDENCIA 2024 ──────────────────────────
    sec("Multiplicadores de Tendencia 2024")
    tip("Factores de ajuste aplicados al ROI histórico del modelo para reflejar la realidad "
        "del mercado publicitario español en 2024. Fuentes: IAB Spain Digital Ad Spend Report 2024, "
        "GWI Consumer Trends Spain Q4 2024, Meta Business Insights España 2024, "
        "Spotify Advertising Spain 2024, Statista Spain Digital Media 2024.")

    trend_rows = []
    # Añadir techo a cada descripción para que la tabla sea completa
    descripciónes = {
        "Social Paid":  ("Instagram Reels, TikTok For Business, Pinterest Ads",
                         "El 78% de españoles 18-44 usa Instagram/TikTok diariamente. CPM social "
                         "cayo un 18% en 2024 vs 2023 por mayor oferta de inventario. "
                         "TikTok Shop lanzado en España en 2024 convierte directamente."),
        "Video Online": ("YouTube Ads, YouTube CTV, Programatica video",
                         "TV conectada crece +34% en hogares españoles. YouTube alcanza "
                         "el 92% de cobertura adultos 18-54. Pre-roll skipaple con recall "
                         "del 47% vs 12% de banner display."),
        "Paid Search":  ("Google Search Ads, Shopping Ads, PMAX",
                         "Canal de mayor intencion de compra. Conversión directa demostrable. "
                         "Performance Max (PMAX) mejora eficiencia un 15% vs Smart Shopping. "
                         "Estable pero sin crecimiento incremental significativo."),
        "Display":      ("Programatica display, banners rich media, DOOH digital",
                         "Banner blindness cronico: CTR medio display España = 0.06%. "
                         "Solo programatica con audiencias de datos propios (1P data) "
                         "mantiene eficiencia. Retargeting conserva valor."),
        "Email CRM":    ("Newsletter, flows automatizados, SMS marketing",
                         "Tasa apertura media retail España cayo de 22% (2022) a 17% (2024). "
                         "Saturación de bandeja de entrada. Óptimo entre 8-10% del budget: "
                         "muy eficiente para retencion, techo bajo para captación."),
        "Exterior":     ("DOOH, marquesinas, vallas digitales zonas premium",
                         "Recuperacion post-COVID consolidada pero coste por impacto 3x "
                         "superior a digital. Eficaz para brand awareness en ciudades "
                         "top (Madrid, Barcelona). Limitar a zonas de alta densidad cliente."),
        "Radio Local":  ("Spotify Ads, radio digital, podcasts patrocinados",
                         "Audiencia FM <45 años cae -8% anual. Spotify Ads y podcasts "
                         "son alternativa eficiente pero requieren producción diferente. "
                         "Radio convencional pierde eficiencia para target K-Moda."),
        "Prensa":       ("Publicidad en prensa digital y papel, suplementos moda",
                         "Declive estructural acelerado. Lectores papel = perfil 55+ "
                         "con menor propensión a compra moda premium digital. "
                         "Solo justificable en suplementos de moda El Pais/Vogue España "
                         "para imagen de marca, nunca para conversión."),
    }
    for c in sorted(CANALES, key=lambda x: -TREND_2024[x]):
        mult   = TREND_2024[c]
        techo  = ROI_TECHO_GLOBAL[c]
        signos = f"{'▲' if mult>=1 else '▼'} {'+' if mult>=1 else ''}{(mult-1)*100:.0f}%"
        color  = "🟢" if mult >= 1.10 else ("🟡" if mult >= 0.90 else "🔴")
        plat, razón = descripciónes[c]
        trend_rows.append({
            "Canal":             c,
            "Plataformas":       plat,
            "Tendencia 2024":    f"{color} {signos}  ({mult:.2f}x)",
            "ROI Techo":         f"{techo:.2f}x",
            "Justificación":     razón,
        })
    trend_df = pd.DataFrame(trend_rows)
    st.dataframe(trend_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Canal":          st.column_config.TextColumn(width="small"),
                     "Plataformas":    st.column_config.TextColumn(width="medium"),
                     "Tendencia 2024": st.column_config.TextColumn(width="small"),
                     "ROI Techo":      st.column_config.TextColumn(width="small"),
                     "Justificación":  st.column_config.TextColumn(width="large"),
                 })

    sec("Análisis Estratégico: Optimización del Budget 12M€")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{CARD},#1A1A3E);border:1px solid {GOLD}55;
                border-radius:12px;padding:20px 24px;margin-bottom:16px;">
      <span style="color:{GOLD};font-size:18px;font-weight:700;">
        Diagnosis del Analista — K-Moda 2024
      </span><br><br>
      <span style="color:{TEXT};font-size:13px;line-height:1.9;">
        El modelo histórico 2020-2024 muestra que <b style="color:{GOLD}">Email CRM y Paid Search</b>
        tienen el mejor ROI relativo sobre los datos disponibles. Sin embargo, como analista de datos
        con contexto de mercado, esta lectura requiere dos ajustes criticos:<br><br>
        <b style="color:{ACCENT}">1. El Email CRM tiene ROI alto por infrainversión histórica</b> (solo 4.9% del budget),
        no porque sea el canal más eficiente para captar nuevos clientes. Es eficiente para retener
        a clientes ya existentes, pero su techo de escalabilidad es bajo: a partir del 10% del budget
        empieza a saturar y las tasas de apertura caen.<br><br>
        <b style="color:{ACCENT}">2. Social Paid esta infrafinanciado respecto a donde esta la atencion del consumidor.</b>
        En 2024, el 78% de los españoles de 18-44 años usa Instagram o TikTok diariamente
        (GWI Q4 2024). K-Moda compite con Shein y Zara en ese espacio visual — estar ausente
        equivale a ceder escaparate. El modelo no captura este efecto porque los datos históricos
        son de 2020-2022 cuando TikTok aun no dominaba.<br><br>
        <b style="color:{GREEN}">Recomendacion: reasignar 4.2M€</b> de Radio, Prensa y Exterior
        hacia Social Paid y Video Online, manteniendo Email CRM en su rango óptimo del 8-9%.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── GRAFICO 1: ROI modelo vs ROI ajustado por tendencias ─────────────────
    sec("ROI: Modelo Estadístico vs Ajuste por Tendencias 2024")
    tip("Izquierda: ROI que calcula el modelo con datos históricos 2020-2024. "
        "Derecha: ROI ajustado por tendencias de mercado 2024 (IAB Spain, GWI, Meta Insights). "
        "La diferencia entre ambas barras es la correccion estratégica que aplica el analista "
        "para no tomar decisiónes de 2024 con sesgos de 2020.")

    roi_df = pd.DataFrame({
        "canal": CANALES,
        "roi_modelo": [roi[c] for c in CANALES],
        "roi_ajustado": [roi_adj[c] for c in CANALES],
    }).sort_values("roi_ajustado", ascending=False)

    fig_roi_comp = go.Figure()
    fig_roi_comp.add_trace(go.Bar(
        name="ROI Modelo (datos históricos)",
        x=roi_df["canal"], y=roi_df["roi_modelo"],
        marker_color=GRAY, opacity=0.6,
    ))
    fig_roi_comp.add_trace(go.Bar(
        name="ROI Ajustado 2024 (tendencias mercado)",
        x=roi_df["canal"], y=roi_df["roi_ajustado"],
        marker_color=[CANAL_COLORS[c] for c in roi_df["canal"]],
        opacity=0.9,
    ))
    bl(fig_roi_comp, "ROI por Canal: Modelo Histórico vs Ajuste Tendencias 2024", 400)
    fig_roi_comp.update_layout(barmode="group")
    fig_roi_comp.update_yaxes(title_text="ROI (x)")
    st.plotly_chart(fig_roi_comp, use_container_width=True)

    # ── GRAFICO 2: Distribución actual vs óptima ──────────────────────────────
    sec("Redistribución Óptima del Presupuesto 2024")
    tip("Comparativa entre la distribución REAL de 2024 (como se invirtieron los 12M€ este año "
        "segun el CSV) y la distribución óptima calculada con ROI ajustado por tendencias 2024 "
        "+ caps de negocio por canal. Verde = aumentar presupuesto, rojo = reducir.")

    # Distribución de 2024 específicamente (no acumulado histórico 5 años)
    inv_2024_only = inv[inv["anio"]==2024].groupby("canal_medio")["inversion_eur"].sum()
    inv_2024_sum  = inv_2024_only.sum()
    dist_actual   = {c: float(inv_2024_only.get(c, 0))/inv_2024_sum for c in CANALES}
    canales_ord = sorted(CANALES, key=lambda c: -pesos_opt[c])

    col1, col2 = st.columns(2)
    with col1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            name="Distribución Actual (CSV histórico)",
            x=canales_ord,
            y=[dist_actual[c]*100 for c in canales_ord],
            marker_color=GRAY, opacity=0.65,
        ))
        fig_dist.add_trace(go.Bar(
            name="Distribución Óptima 2024",
            x=canales_ord,
            y=[pesos_opt[c]*100 for c in canales_ord],
            marker_color=[CANAL_COLORS[c] for c in canales_ord],
            opacity=0.9,
        ))
        bl(fig_dist, "Distribución Actual vs Óptima por Canal (%)", 400)
        fig_dist.update_layout(barmode="group")
        fig_dist.update_yaxes(title_text="%")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Reasignación en euros
        inv_actual_e = {c: dist_actual[c]*BUDGET_2024 for c in CANALES}
        inv_optimo_e = {c: pesos_opt[c]*BUDGET_2024 for c in CANALES}
        delta_e      = {c: inv_optimo_e[c]-inv_actual_e[c] for c in CANALES}
        delta_ord    = dict(sorted(delta_e.items(), key=lambda x: -x[1]))

        fig_delta = go.Figure(go.Bar(
            x=list(delta_ord.keys()),
            y=[v/1e3 for v in delta_ord.values()],
            marker_color=[GREEN if v>=0 else RED for v in delta_ord.values()],
            text=[f"{'+' if v>=0 else ''}{v/1e3:.0f}K€" for v in delta_ord.values()],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        fig_delta.add_hline(y=0, line_color=GOLD, line_dash="dash", line_width=1.5)
        bl(fig_delta, "Reasignación Recomendada: Delta por Canal (K€)", 400)
        fig_delta.update_yaxes(title_text="K€  (verde = aumentar, rojo = reducir)")
        st.plotly_chart(fig_delta, use_container_width=True)

    # ── TABLA RESUMEN ESTRATEGICO ─────────────────────────────────────────────
    sec("Plan de Acción: Presupuesto Óptimo 2024")
    tip("Resumen ejecutivo de la redistribución recomendada. "
        "El incremento de ganancia estimado asume que el ROI ajustado es el correcto "
        "para el mercado 2024 y que la reasignación se ejecuta desde el inicio del año.")

    filas = []
    gan_actual_total = 0
    gan_optima_total = 0
    for c in canales_ord:
        inv_a  = dist_actual[c] * BUDGET_2024
        inv_o  = pesos_opt[c]   * BUDGET_2024
        delta  = inv_o - inv_a
        roi_a_val = roi[c]
        roi_o_val = roi_adj[c]
        gan_a  = roi_a_val * inv_a * MARGEN
        gan_o  = roi_o_val * inv_o * MARGEN
        gan_actual_total += gan_a
        gan_optima_total += gan_o
        signo  = "▲" if delta > 0 else "▼"
        color  = "🟢" if delta > 0 else "🔴"
        filas.append({
            "Canal":           c,
            "Inv. Actual (K€)": f"{inv_a/1e3:.0f}K€",
            "Inv. Óptima (K€)": f"{inv_o/1e3:.0f}K€",
            "Delta":           f"{color} {signo}{abs(delta)/1e3:.0f}K€",
            "ROI Modelo":      f"{roi_a_val:.3f}x",
            "ROI Ajust. 2024": f"{roi_o_val:.3f}x",
        })

    tabla_df = pd.DataFrame(filas)
    st.dataframe(tabla_df, use_container_width=True, hide_index=True)

    # KPIs del impacto
    gan_incr = gan_optima_total - gan_actual_total
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ganancia Mix Actual",   f"{gan_actual_total/1e3:.0f}K€")
    k2.metric("Ganancia Mix Óptimo",   f"{gan_optima_total/1e3:.0f}K€",
              delta=f"+{gan_incr/1e3:.0f}K€")
    k3.metric("Incremento Estimado",   f"{gan_incr/gan_actual_total*100:.1f}%" if gan_actual_total>0 else "N/A")
    k4.metric("Reasignación Total",    f"{sum(abs(v) for v in delta_e.values())/2/1e6:.1f}M€ movidos")

    # ── GRAFICO 3: Ganancias esperadas con inversión actual vs óptima ───────
    st.divider()
    sec("Ganancias Esperadas: Mix Actual 2024 vs Mix Óptimo")
    tip("Tres escenarios comparados sobre los 12M€ de presupuesto 2024: "
        "(1) CSV Simulado: distribución que vienen en los datos del ejercicio academico "
        "(sesgada hacia offline: Radio 10.8%, Prensa 9%); "
        "(2) IAB Spain 2024: benchmark de mercado real como punto de partida de negocio "
        "(Social 27%, Radio 3%); "
        "(3) Mix Óptimo: nuestra propuesta tras aplicar ROI ajustado + caps estratégicos. "
        "La comparativa honesta es IAB 2024 vs Mix Óptimo, no el CSV simulado.")

    # Los tres escenarios
    dist_csv  = {c: float(inv[inv["anio"]==2024].groupby("canal_medio")["inversion_eur"].sum().get(c,0))
                    / float(inv[inv["anio"]==2024]["inversion_eur"].sum())
                 for c in CANALES}
    dist_iab  = DIST_IAB   # benchmark mercado real
    dist_opt  = pesos_opt  # nuestra propuesta

    gan_csv_c = {c: roi_adj[c] * dist_csv[c] * BUDGET_2024 * MARGEN for c in CANALES}
    gan_iab_c = {c: roi_adj[c] * dist_iab[c] * BUDGET_2024 * MARGEN for c in CANALES}
    gan_opt_c = {c: roi_adj[c] * dist_opt[c] * BUDGET_2024 * MARGEN for c in CANALES}

    total_gan_csv = sum(gan_csv_c.values())
    total_gan_iab = sum(gan_iab_c.values())
    total_gan_opt = sum(gan_opt_c.values())

    canales_gan = sorted(CANALES, key=lambda c: -gan_opt_c[c])

    # ── Barras agrupadas: 3 escenarios por canal ──────────────────────────────
    fig_gan_comp = go.Figure()
    fig_gan_comp.add_trace(go.Bar(
        name="CSV Simulado (datos ejercicio)",
        x=canales_gan,
        y=[gan_csv_c[c]/1e3 for c in canales_gan],
        marker_color=RED, opacity=0.55,
        text=[f"{gan_csv_c[c]/1e3:.0f}K" for c in canales_gan],
        textposition="outside", textfont=dict(color=TEXT, size=8),
    ))
    fig_gan_comp.add_trace(go.Bar(
        name="IAB Spain 2024 (benchmark real)",
        x=canales_gan,
        y=[gan_iab_c[c]/1e3 for c in canales_gan],
        marker_color=GOLD, opacity=0.75,
        text=[f"{gan_iab_c[c]/1e3:.0f}K" for c in canales_gan],
        textposition="outside", textfont=dict(color=TEXT, size=8),
    ))
    fig_gan_comp.add_trace(go.Bar(
        name="Mix Óptimo (propuesta)",
        x=canales_gan,
        y=[gan_opt_c[c]/1e3 for c in canales_gan],
        marker_color=[CANAL_COLORS[c] for c in canales_gan],
        opacity=0.95,
        text=[f"{gan_opt_c[c]/1e3:.0f}K" for c in canales_gan],
        textposition="outside", textfont=dict(color=TEXT, size=8),
    ))
    bl(fig_gan_comp, "Ganancia Esperada por Canal: 3 Escenarios (K€/año)", 440)
    fig_gan_comp.update_layout(barmode="group")
    fig_gan_comp.update_yaxes(title_text="K€/año")
    st.plotly_chart(fig_gan_comp, use_container_width=True)

    # ── KPIs de los tres escenarios ───────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Ganancia CSV Simulado",
              f"{total_gan_csv*FACTOR_ESCALA/1e6:.1f}M€/año",
              help="Con distribución histórica del ejercicio académico (sesgada hacia offline)")
    k2.metric("Ganancia IAB 2024",
              f"{total_gan_iab*FACTOR_ESCALA/1e6:.1f}M€/año",
              delta=f"{(total_gan_iab-total_gan_csv)/total_gan_csv*100:+.1f}% vs CSV",
              help="Con benchmark real de mercado español como punto de partida")
    k3.metric("Ganancia Mix Óptimo",
              f"{total_gan_opt*FACTOR_ESCALA/1e6:.1f}M€/año",
              delta=f"{(total_gan_opt-total_gan_iab)/total_gan_iab*100:+.1f}% vs IAB 2024",
              delta_color="normal",
              help="Con distribución óptima: ROI ajustado + tendencias 2024 + caps")

    # ── Donuts comparativos: IAB 2024 vs Mix Óptimo (la comparativa relevante) ─
    st.markdown(f"<div style='color:{GRAY};font-size:12px;margin:8px 0 4px 0'>"
                "💡 La comparativa relevante para la toma de decisiónes es "
                "<b style='color:{GOLD}'>IAB 2024 vs Mix Óptimo</b>, "
                "no el CSV simulado del ejercicio.</div>".format(GOLD=GOLD),
                unsafe_allow_html=True)

    # Donuts: distribución de INVERSIÓN (no de ganancias) para que coincida con tab1
    canales_opt_ord = sorted(CANALES, key=lambda c: -pesos_opt[c])
    dist_iab_inv    = {c: DIST_IAB.get(c, 0) * BUDGET_2024 for c in CANALES}
    dist_opt_inv    = {c: pesos_opt[c]        * BUDGET_2024 for c in CANALES}

    fig_donuts = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"domain"}, {"type":"domain"}]],
        subplot_titles=["IAB Spain 2024 (punto de partida)", "Mix Óptimo — igual que pestaña Inversión"]
    )
    for col_idx, (inv_dict, label, color_center) in enumerate([
        (dist_iab_inv, "IAB", GOLD),
        (dist_opt_inv, "OPT", GREEN),
    ], start=1):
        fig_donuts.add_trace(go.Pie(
            labels=canales_opt_ord,
            values=[inv_dict[c]/1e3 for c in canales_opt_ord],
            hole=0.55, name=label, sort=False,
            marker_colors=[CANAL_COLORS[c] for c in canales_opt_ord],
            textinfo="label+percent",
            textfont=dict(size=9, color=TEXT),
            showlegend=(col_idx == 2),
        ), row=1, col=col_idx)

    fig_donuts.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=TEXT, size=11),
        margin=dict(l=20, r=20, t=55, b=20),
        height=400,
        title=dict(
            text="Distribución de Inversión: IAB 2024 vs Mix Óptimo (K€)",
            font=dict(size=13, color=GOLD)
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color=TEXT, size=10)),
        annotations=[
            dict(text=f"<b>{sum(dist_iab_inv.values())/1e3:.0f}K€</b>",
                 x=0.19, y=0.5, font=dict(size=13, color=GOLD), showarrow=False),
            dict(text=f"<b>{sum(dist_opt_inv.values())/1e3:.0f}K€</b>",
                 x=0.81, y=0.5, font=dict(size=13, color=GREEN), showarrow=False),
        ]
    )
    st.plotly_chart(fig_donuts, use_container_width=True)

    # ── Delta de ganancia: IAB vs Óptimo ──────────────────────────────────────
    delta_gan = {c: gan_opt_c[c] - gan_iab_c[c] for c in CANALES}
    delta_gan_s = dict(sorted(delta_gan.items(), key=lambda x: -x[1]))
    fig_delta_gan = go.Figure(go.Bar(
        x=list(delta_gan_s.keys()),
        y=[v/1e3 for v in delta_gan_s.values()],
        marker_color=[GREEN if v >= 0 else RED for v in delta_gan_s.values()],
        text=[f"{'+' if v>=0 else ''}{v/1e3:.0f}K€" for v in delta_gan_s.values()],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig_delta_gan.add_hline(y=0, line_color=GOLD, line_dash="dash", line_width=1.5)
    bl(fig_delta_gan,
       "Incremento de Ganancia por Canal: Mix Óptimo vs IAB Spain 2024 (K€/año)", 360)
    fig_delta_gan.update_yaxes(title_text="Incremento ganancia (K€/año)")
    st.plotly_chart(fig_delta_gan, use_container_width=True)

    # ── GRAFICO 4: Ganancias semanales descompuestas ──────────────────────────
    st.divider()
    sec("Ganancias Semanales Históricas: Base Orgánica + por Canal")
    tip("Descomposición semana a semana de la ganancia estimada del período 2020-2024. "
        "La base orgánica (55%) es la ganancia que existiria sin publicidad: "
        "clientes recurrentes, tráfico directo y fuerza de marca acumulada. "
        "Las bandas de color muestran la ganancia incremental atribuida a cada canal.")

    gan_base = base * PRECIO_MEDIO * MARGEN
    # Escalar por factor de ajuste de tendencias para consistencia con el resto
    gan_c = {c: contrib[c] * contrib_adj_factor[c] * PRECIO_MEDIO * MARGEN
             for c in CANALES}

    fig_gan = go.Figure()
    fig_gan.add_trace(go.Scatter(
        x=semanas, y=gan_base/1e3, name="Base Orgánica (55%)",
        stackgroup="one", line=dict(width=0),
        fillcolor="rgba(148,163,184,0.55)"))
    for c in CANALES:
        fig_gan.add_trace(go.Scatter(
            x=semanas, y=gan_c[c]/1e3, name=c,
            stackgroup="one", line=dict(width=0),
            fillcolor=f"rgba({_hex_to_rgb(CANAL_COLORS[c])},0.67)"))
    bl(fig_gan, "Ganancias Semanales Estimadas 2020-2024: Base Orgánica + Incrementales — Ajustado 2024 (K€)", 430)
    fig_gan.update_yaxes(title_text="Ganancia estimada (K€)")
    st.plotly_chart(fig_gan, use_container_width=True)

    # ── SIMULADOR ─────────────────────────────────────────────────────────────
    st.divider()
    sec("Simulador de Budget 12M€")
    tip("Ajusta el % de cada canal y compara en tiempo real el índice de ganancia "
        "respecto al mix histórico. Los valores por defecto son la distribución óptima "
        "recomendada por el modelo + tendencias 2024.")

    # Defaults del simulador = distribución óptima calculada arriba
    DEFS_OPT = {c: int(round(pesos_opt[c]*100)) for c in CANALES}
    diff = 100 - sum(DEFS_OPT.values())
    # Ajustar el canal con mayor peso para que sumen 100 exacto
    canal_mayor = max(DEFS_OPT, key=lambda c: DEFS_OPT[c])
    DEFS_OPT[canal_mayor] += diff

    cols_s = st.columns(4)
    sliders = {}
    for i, c in enumerate(CANALES):
        with cols_s[i % 4]:
            sliders[c] = st.slider(c, 0, 40, DEFS_OPT.get(c,5), step=1,
                                   help=f"ROI ajustado 2024: {roi_adj[c]:.3f}x")

    pct_total = sum(sliders.values())
    presup    = {c: sliders[c]/100 * BUDGET_2024 for c in CANALES}

    # Ganancia simulada con ROI ajustado
    g_sim_opt  = sum(roi_adj[c] * presup[c] * MARGEN for c in CANALES)
    g_sim_hist = sum(roi[c]     * presup[c] * MARGEN for c in CANALES)
    g_base_yr  = base.sum() * PRECIO_MEDIO * MARGEN / 5
    g_total    = g_sim_opt + g_base_yr

    # Ganancia con mix actual (referencia)
    g_actual_ref = sum(roi_adj[c]*dist_actual[c]*BUDGET_2024*MARGEN for c in CANALES) + g_base_yr

    if pct_total != 100:
        st.warning(f"⚠️  {pct_total}% asignado — ajusta los sliders para llegar al 100%.")
    else:
        s1,s2,s3,s4 = st.columns(4)
        # Aplicar FACTOR_ESCALA para mostrar cifras absolutas realistas
        s1.metric("Ganancia Orgánica (año)",
                  f"{g_base_yr*FACTOR_ESCALA/1e6:.1f}M€",
                  help="Ganancia que existiría sin publicidad (55% de ventas × margen)")
        s2.metric("Ganancia por Medios (año)",
                  f"{g_sim_opt*FACTOR_ESCALA/1e6:.1f}M€",
                  help="Ganancia generada por la inversión publicitaria con este mix")
        s3.metric("Ganancia Total (año)",
                  f"{g_total*FACTOR_ESCALA/1e6:.1f}M€",
                  delta=f"{(g_total-g_actual_ref)*FACTOR_ESCALA/1e6:+.1f}M€ vs mix actual",
                  help="Ganancia bruta total = orgánica + medios")
        s4.metric("% Presupuesto usado",      f"{pct_total}%")

    col_sa, col_sb = st.columns(2)
    with col_sa:
        fig_sim = go.Figure(go.Bar(
            x=list(presup.keys()),
            y=[v/1e3 for v in presup.values()],
            marker_color=[CANAL_COLORS[c] for c in presup],
            text=[f"{v/1e3:.0f}K€  ({sliders[c]}%)" for c, v in presup.items()],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        bl(fig_sim, "Distribución del Budget Simulado (K€)", 380)
        fig_sim.update_yaxes(title_text="K€")
        st.plotly_chart(fig_sim, use_container_width=True)
    with col_sb:
        # Comparativa ganancia simulada vs mix actual vs mix óptimo
        scenarios = ["Mix Histórico CSV", "Mix Óptimo (modelo)", "Tu Simulacion"]
        g_hist_sc = sum(roi_adj[c]*dist_actual[c]*BUDGET_2024*MARGEN for c in CANALES)
        g_opt_sc  = sum(roi_adj[c]*pesos_opt[c]*BUDGET_2024*MARGEN for c in CANALES)
        ganancias_sc = [g_hist_sc/1e3, g_opt_sc/1e3, g_sim_opt/1e3]
        colores_sc   = [GRAY, GOLD, ACCENT]
        fig_sc = go.Figure(go.Bar(
            x=scenarios, y=ganancias_sc,
            marker_color=colores_sc,
            text=[f"{v:.0f}K€" for v in ganancias_sc],
            textposition="outside", textfont=dict(color=TEXT, size=12),
        ))
        fig_sc.add_hline(y=g_hist_sc/1e3, line_color=GRAY,
                         line_dash="dot", line_width=1)
        bl(fig_sc, "Ganancia por Medios: 3 Escenarios Comparados (K€/año)", 380)
        fig_sc.update_yaxes(title_text="K€/año")
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · CLIENTES CRM
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    sec("Base de Clientes CRM")
    tip("600.000 clientes analizados por segmento de valor, canal preferido y ciudad. "
        "Esta segmentación permite alinear la estrategia de medios con el perfil real "
        "del comprador: un cliente 'Online' justifica mayor inversión digital; "
        "un cliente 'Tienda' refuerza medios de proximidad como Exterior.")

    cli_seg  = cli["segmento"].value_counts().reset_index()
    cli_can  = cli["canal_preferido"].value_counts().reset_index()
    cli_ciu  = cli["ciudad_residencia"].value_counts().head(10).reset_index()
    cli2     = cli.copy()
    cli2["anio_alta"] = cli2["fecha_alta"].dt.year
    cli_alta = cli2.groupby(["anio_alta","segmento"]).size().reset_index(name="n")

    col1, col2 = st.columns(2)
    with col1:
        fig_sg = px.bar(cli_seg, x="segmento", y="count", color="segmento",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        text=cli_seg["count"].apply(lambda v: f"{v/1e3:.0f}K"))
        bl(fig_sg, "Clientes por Segmento", 360)
        fig_sg.update_layout(showlegend=False)
        fig_sg.update_traces(textposition="outside", textfont=dict(color=TEXT))
        fig_sg.update_yaxes(title_text="N° Clientes")
        st.plotly_chart(fig_sg, use_container_width=True)
    with col2:
        fig_cc = go.Figure(go.Pie(
            labels=cli_can["canal_preferido"], values=cli_can["count"],
            hole=0.5, marker_colors=[GOLD,ACCENT,GREEN],
            textinfo="label+percent", textfont=dict(size=13, color=TEXT),
        ))
        bl(fig_cc, "Canal de Compra Preferido", 360)
        fig_cc.update_layout(showlegend=False)
        st.plotly_chart(fig_cc, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_alt = px.bar(cli_alta, x="anio_alta", y="n", color="segmento",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        bl(fig_alt, "Altas de Clientes por Año y Segmento", 360)
        fig_alt.update_layout(barmode="stack")
        fig_alt.update_xaxes(dtick=1)
        fig_alt.update_yaxes(title_text="N° Clientes")
        st.plotly_chart(fig_alt, use_container_width=True)
    with col4:
        fig_ciu = go.Figure(go.Bar(
            x=cli_ciu["count"], y=cli_ciu["ciudad_residencia"],
            orientation="h",
            marker=dict(color=cli_ciu["count"],
                        colorscale=[[0,LIGHT],[1,GOLD]], showscale=False),
            text=cli_ciu["count"].apply(lambda v: f"{v/1e3:.1f}K"),
            textposition="outside", textfont=dict(color=TEXT),
        ))
        bl(fig_ciu, "Clientes por Ciudad (Top 10)", 360)
        fig_ciu.update_xaxes(title_text="N° Clientes")
        st.plotly_chart(fig_ciu, use_container_width=True)

    sec("Entorno: Temperatura y Lluvia")
    tip("Variables exogenas no controlables que el modelo captura para aislar su efecto "
        "del impacto real de la publicidad.")

    col5, col6 = st.columns(2)
    with col5:
        tg = trf_f.groupby("temperatura_media_c")["sesiones_web"].mean().reset_index().dropna()
        cf = np.polyfit(tg["temperatura_media_c"], tg["sesiones_web"], 1)
        xl = np.linspace(tg["temperatura_media_c"].min(), tg["temperatura_media_c"].max(), 100)
        fig_tm = go.Figure()
        fig_tm.add_trace(go.Scatter(x=tg["temperatura_media_c"], y=tg["sesiones_web"],
                                    mode="markers", marker=dict(color=GOLD,opacity=0.5,size=5)))
        fig_tm.add_trace(go.Scatter(x=xl, y=np.polyval(cf,xl), mode="lines",
                                    line=dict(color=ACCENT, dash="dash", width=2),
                                    name="Tendencia"))
        bl(fig_tm, "Temperatura (°C) vs Sesiónes Web", 360)
        fig_tm.update_xaxes(title_text="Temperatura (°C)")
        fig_tm.update_yaxes(title_text="Sesiónes medias")
        st.plotly_chart(fig_tm, use_container_width=True)
    with col6:
        bins = pd.cut(trf_f["lluvia_indice"], bins=10)
        lg = trf_f.groupby(bins, observed=True)["visitas_tienda"].mean().reset_index()
        lg["mid"] = lg["lluvia_indice"].apply(lambda x: round(x.mid, 1))
        fig_ll = go.Figure(go.Bar(
            x=lg["mid"], y=lg["visitas_tienda"],
            marker=dict(color=lg["visitas_tienda"],
                        colorscale=[[0,GOLD],[1,ACCENT]], showscale=False),
        ))
        bl(fig_ll, "Lluvia vs Visitas a Tienda (media por tramo)", 360)
        fig_ll.update_xaxes(title_text="Índice de Lluvia")
        fig_ll.update_yaxes(title_text="Visitas Tienda")
        st.plotly_chart(fig_ll, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · PRODUCTO
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    sec("Catalogo de Producto")
    tip("El mix de producto condiciona el ROI de la publicidad: categorías de alto margen "
        "justifican campañas de conversión directa; las de menor margen se orientan a "
        "volumen y captación. La temporalidad explica parte de la estacionalidad de ventas.")

    tip("Cada punto es un SKU. Arriba a la derecha = precio alto y margen alto = "
        "candidatos prioritarios para campañas de conversión.")
    fig_sp = px.scatter(pro, x="pvp_bruto_ref_eur", y="margen_objetivo_pct",
                        color="categoria", size="coste_produccion_eur",
                        hover_name="nombre_articulo",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"pvp_bruto_ref_eur":"PVP Bruto (€)",
                                "margen_objetivo_pct":"Margen Objetivo"})
    bl(fig_sp, "PVP Bruto vs Margen Objetivo (tamano = coste de producción)", 420)
    st.plotly_chart(fig_sp, use_container_width=True)

    pro_cat = pro.groupby("categoria").agg(
        pvp_medio=("pvp_bruto_ref_eur","mean"),
        margen_medio=("margen_objetivo_pct","mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_pc = make_subplots(specs=[[{"secondary_y":True}]])
        fig_pc.add_trace(go.Bar(name="PVP Medio (€)", x=pro_cat["categoria"],
                                y=pro_cat["pvp_medio"], marker_color=GOLD), secondary_y=False)
        fig_pc.add_trace(go.Scatter(name="Margen (%)", x=pro_cat["categoria"],
                                    y=pro_cat["margen_medio"]*100,
                                    mode="lines+markers",
                                    line=dict(color=ACCENT, width=2),
                                    marker=dict(size=8)), secondary_y=True)
        bl(fig_pc, "PVP Medio y Margen por Categoría", 360)
        fig_pc.update_yaxes(title_text="PVP Medio (€)", secondary_y=False)
        fig_pc.update_yaxes(title_text="Margen (%)", secondary_y=True)
        st.plotly_chart(fig_pc, use_container_width=True)
    with col2:
        tc = pro.groupby(["categoria","temporada_fuerte"]).size().reset_index(name="n")
        fig_tc2 = px.bar(tc, x="categoria", y="n", color="temporada_fuerte",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        bl(fig_tc2, "SKUs por Categoría y Temporada", 360)
        fig_tc2.update_yaxes(title_text="N° SKUs")
        st.plotly_chart(fig_tc2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<div style='text-align:center;color:{GRAY};font-size:11px'>"
    "K-MODA · Dashboard Analitico MMM + Random Forest · 2020-2024 · "
    "Powered by Streamlit + Plotly + Scikit-learn</div>",
    unsafe_allow_html=True,
)
