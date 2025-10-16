# h2_simulator_v2_6.py — Header-Icons entfernt • Legenden transparent • H2-Preis ab 1.50 €/kg
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Bosch H2-Landschaft Simulator — V2.6",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BOSCH = {
    "dark_blue":  "#005691",
    "light_green":"#78BE20",
    "turquoise":  "#00A8B0",
    "dark_gray":  "#525F6B",
    "bg":         "#F7F8FA",
}

pio.templates["bosch"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Roboto, Arial, sans-serif", color=BOSCH["dark_gray"]),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=[BOSCH["dark_blue"], BOSCH["light_green"], BOSCH["turquoise"], "#B90276", "#50237F", "#008ECF", "#E20015", "#006249", "#BFC0C2"],
        xaxis=dict(gridcolor="#ECEFF1", zerolinecolor="#ECEFF1"),
        yaxis=dict(gridcolor="#ECEFF1", zerolinecolor="#ECEFF1"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "bosch"

PDF_DEFAULTS = {
    "Deutschland": {"strompreis": 0.18, "co2_intensitaet": 350, "h2_verkaufspreis": 12.85, "dieselpreis": 1.57, "bev_strompreis": 0.18},
    "EU":          {"strompreis": 0.1899, "co2_intensitaet": 230, "h2_verkaufspreis": 12.85, "dieselpreis": 1.50, "bev_strompreis": 0.1899},
    "USA":         {"strompreis": 0.128, "co2_intensitaet": 450, "h2_verkaufspreis": 31.79, "dieselpreis": 0.902, "bev_strompreis": 0.128},
    "China":       {"strompreis": 0.089, "co2_intensitaet": 550, "h2_verkaufspreis": 3.90,  "dieselpreis": 0.898, "bev_strompreis": 0.089},
}

def _rng(val, f1=0.5, f2=1.6, lo=None, hi=None):
    a = max(lo if lo is not None else -1e9, val*f1)
    b = min(hi if hi is not None else  1e9, val*f2)
    return (round(a, 4), round(b, 4))

st.title("⚡ Bosch H2-Landschaft Simulator — V2.6")
st.caption("Icons im Header entfernt • Legenden ohne Hintergrund • H₂-Preis-Slider ab 1.50 €/kg.")

st.sidebar.header("🎛️ Simulationsparameter")
regionen = list(PDF_DEFAULTS.keys())
selected_region = st.sidebar.selectbox("🌍 Region", options=regionen, index=0)
defs = PDF_DEFAULTS[selected_region]

strompreis = st.sidebar.slider("Industrie-Strompreis (€/kWh)", *_rng(defs["bev_strompreis"], 0.5, 1.6, 0.05, 0.8), value=float(defs["bev_strompreis"]), step=0.005)
h2_min = 1.5
h2_max = max(_rng(defs["h2_verkaufspreis"], 0.6, 1.6, 1.5, 40.0)[1], 4.0)
h2_preis    = st.sidebar.slider("H₂-Preis (€/kg)", h2_min, h2_max, float(max(defs["h2_verkaufspreis"], h2_min)), 0.1)
dieselpreis = st.sidebar.slider("Dieselpreis (€/L)", *_rng(defs["dieselpreis"], 0.6, 1.6, 0.6, 3.5), value=float(defs["dieselpreis"]), step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Elektrolyse-Setup")
num_stacks = st.sidebar.slider("Anzahl 1.25‑MW‑Stacks", 1, 100, 10, 1)
volllaststunden = st.sidebar.slider("Volllaststunden (h/Jahr)", 2000, 8000, 5000, 100)
investitionsdauer = st.sidebar.slider("Investitionsdauer (Jahre)", 10, 20, 15, 1)
wartungskosten_prozent = st.sidebar.slider("Wartungskosten (% von CAPEX)", 1.0, 5.0, 2.5, 0.5)
co2_intensitaet = st.sidebar.slider("CO₂-Intensität Strom (g/kWh)", 20, 800, int(defs["co2_intensitaet"]), 10)

# Fixed parameters
P1_stack_leistung = 1.25  # MW
P2_h2_produktion = 22.9   # kg/h
P3_spez_stromverbrauch = 53.0  # kWh/kg H2
P4_capex_pro_mw = 1_400_000    # €/MW

# Mobility parameters
P10_h2_lkw_verbrauch = 7.5
P11_diesel_lkw_verbrauch = 31.5
P12_bev_lkw_verbrauch = 130.0
P13_co2_emission_diesel = 3.15

def compute(num_stacks, strompreis, volllaststunden, investitionsdauer, wartungskosten_prozent, h2_preis, dieselpreis, co2_intensitaet):
    L_ges = num_stacks * P1_stack_leistung
    M_h2_jahr = num_stacks * P2_h2_produktion * volllaststunden
    E_strom_jahr = M_h2_jahr * P3_spez_stromverbrauch
    capex = L_ges * P4_capex_pro_mw
    c_capex = capex / investitionsdauer
    c_strom = E_strom_jahr * strompreis
    c_wart = capex * (wartungskosten_prozent / 100.0)
    opex = c_strom + c_wart
    lcoh = (c_capex + opex) / M_h2_jahr
    e_h2 = (E_strom_jahr * co2_intensitaet) / 1000.0
    e_grau = M_h2_jahr * 10.0
    co2_save = e_grau - e_h2
    k_h2 = h2_preis * P10_h2_lkw_verbrauch
    k_diesel = dieselpreis * P11_diesel_lkw_verbrauch
    k_bev = strompreis * P12_bev_lkw_verbrauch
    co2_h2 = (P10_h2_lkw_verbrauch * P3_spez_stromverbrauch / 100.0) * co2_intensitaet / 1000.0
    co2_diesel = P11_diesel_lkw_verbrauch * P13_co2_emission_diesel
    co2_bev = (P12_bev_lkw_verbrauch * co2_intensitaet) / 1000.0
    return dict(L_ges=L_ges, M_h2_jahr=M_h2_jahr, E_strom_jahr=E_strom_jahr, capex=capex,
                c_capex=c_capex, c_strom=c_strom, c_wart=c_wart, opex=opex, lcoh=lcoh,
                e_h2=e_h2, e_grau=e_grau, co2_save=co2_save,
                k_h2=k_h2, k_diesel=k_diesel, k_bev=k_bev,
                co2_h2=co2_h2, co2_diesel=co2_diesel, co2_bev=co2_bev)

res = compute(num_stacks, strompreis, volllaststunden, investitionsdauer, wartungskosten_prozent, h2_preis, dieselpreis, co2_intensitaet)

# KPIs
k1, k2, k3 = st.columns(3)
with k1: st.metric("LCOH", f"{res['lcoh']:.2f} €/kg", f"{res['lcoh']-10:.2f} vs Ziel 10")
with k2: st.metric("H₂-Produktion", f"{res['M_h2_jahr']/1000:.1f} t/a", f"{res['M_h2_jahr']/1000/365:.1f} t/d")
with k3: st.metric("CO₂‑Einsparung vs Grau", f"{res['co2_save']/1000:.1f} t/a", f"{(res['co2_save']/res['e_grau'])*100:.1f}%")

st.markdown("---")

# H2-Produktion
st.header("📊 Analyse der H₂‑Produktion")
a,b = st.columns([2,1])
with a:
    st.subheader("Kostenverteilung (€/Jahr)")
    df_cost = pd.DataFrame({"Kategorie":["CAPEX (ann.)","Strom","Wartung"], "Kosten":[res["c_capex"],res["c_strom"],res["c_wart"]]})
    fig = px.pie(df_cost, values="Kosten", names="Kategorie", hole=0.35, template="bosch")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
with b:
    st.write("")

st.subheader("LCOH‑Breakdown (€/kg H₂)")
df_lcoh = pd.DataFrame({
    "Komponente":["CAPEX","Strom","Wartung"],
    "€/kg":[res["c_capex"]/res["M_h2_jahr"], res["c_strom"]/res["M_h2_jahr"], res["c_wart"]/res["M_h2_jahr"]]
})
bar = go.Figure([go.Bar(x=df_lcoh["Komponente"], y=df_lcoh["€/kg"], text=[f"{v:.2f} €" for v in df_lcoh["€/kg"]], textposition="auto")])
bar.update_layout(yaxis_title="€/kg H₂", template="bosch", height=380, showlegend=True,
                  legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
st.plotly_chart(bar, use_container_width=True)

st.markdown("---")

# Mobilität
st.header("🚛 Mobilitätsvergleich (€/100 km & kg CO₂/100 km)")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Kosten pro 100 km (€)")
    df_mc = pd.DataFrame({"Antriebsart":["H₂-LKW","Diesel-LKW","BEV-LKW"], "Kosten":[res["k_h2"],res["k_diesel"],res["k_bev"]]})
    figc = go.Figure([go.Bar(x=df_mc["Antriebsart"], y=df_mc["Kosten"], text=[f"{v:.2f} €" for v in df_mc["Kosten"]], textposition="auto")])
    figc.update_layout(yaxis_title="€/100 km", template="bosch", height=360, showlegend=False,
                       legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    st.plotly_chart(figc, use_container_width=True)
with c2:
    st.subheader("CO₂ pro 100 km (kg)")
    df_mco2 = pd.DataFrame({"Antriebsart":["H₂-LKW","Diesel-LKW","BEV-LKW"], "CO₂":[res["co2_h2"],res["co2_diesel"],res["co2_bev"]]})
    figm = go.Figure([go.Bar(x=df_mco2["Antriebsart"], y=df_mco2["CO₂"], text=[f"{v:.2f}" for v in df_mco2["CO₂"]], textposition="auto")])
    figm.update_layout(yaxis_title="kg CO₂/100 km", template="bosch", height=360, showlegend=False,
                       legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    st.plotly_chart(figm, use_container_width=True)

st.markdown("---")

# Energiemodule (vereinfacht aus V2.5 übernommen)
st.header("🔌 Energiemodule: Wind • Solar • AKW — Äquivalenzen")
col_w, col_s, col_n = st.columns(3)
with col_w:
    st.subheader("Wind")
    wind_rating = st.slider("Nennleistung pro Turbine (MW)", 4.0, 12.0, 5.0, 0.1)
    wind_cf = st.slider("Auslastung (Capacity Factor)", 0.15, 0.6, 0.30, 0.01)
    wind_TWh_per_turbine = wind_rating * wind_cf * 8760.0 / 1e6
    st.metric("Erzeugung pro Turbine", f"{wind_TWh_per_turbine*1e3:.1f} GWh/a")
with col_s:
    st.subheader("Solar")
    solar_area = st.slider("Fläche Solarpark (km²)", 0.5, 10.0, 3.0, 0.1)
    solar_w_per_km2 = 33.3
    solar_cf = st.slider("Auslastung (%)", 5, 25, 12, 1) / 100.0
    solar_rating = solar_area * solar_w_per_km2
    solar_TWh = solar_rating * solar_cf * 8760.0 / 1e6
    st.metric("Erzeugung Solarpark", f"{solar_TWh*1e3:.1f} GWh/a")
with col_n:
    st.subheader("AKW")
    akw_rating = st.slider("AKW‑Leistung (MW)", 500, 1600, 1000, 50)
    akw_cf = st.slider("Auslastung AKW", 0.7, 0.98, 0.90, 0.01)
    akw_TWh = akw_rating * akw_cf * 8760.0 / 1e6
    st.metric("Erzeugung AKW", f"{akw_TWh:.2f} TWh/a")

st.subheader("Wie viele Windräder/Solarparks ersetzen ein AKW?")
c_w2, c_s2 = st.columns(2)
with c_w2:
    n_wind = akw_TWh / max(wind_TWh_per_turbine, 1e-9)
    st.metric("Windräder ≈", f"{n_wind:,.0f}")
with c_s2:
    n_solar = akw_TWh / max(solar_TWh, 1e-9)
    st.metric("Solarparks (mit obiger Größe) ≈", f"{n_solar:,.1f}")

st.markdown("---")

# Korrelationen
st.header("📈 Korrelationen & Sensitivitäten")
cx, cy, cz = st.columns(3)
rng_strom = cx.slider("Range Strompreis (€/kWh)", 0.05, 0.80, (max(0.05, strompreis*0.7), min(0.80, strompreis*1.3)), 0.01)
rng_vls   = cy.slider("Range Volllaststunden (h/a)", 2000, 8000, (max(2000, volllaststunden-1000), min(8000, volllaststunden+1000)), 100)
n_pts     = cz.slider("Stichproben", 20, 200, 80, 10)

sr = np.random.uniform(rng_strom[0], rng_strom[1], n_pts)
vr = np.random.uniform(rng_vls[0], rng_vls[1], n_pts)

rows = []
for s,v in zip(sr,vr):
    r = compute(num_stacks, s, int(v), investitionsdauer, wartungskosten_prozent, h2_preis, dieselpreis, co2_intensitaet)
    rows.append({
        "Strompreis_€/kWh": float(s),
        "Volllaststunden": int(v),
        "LCOH_€/kg": r["lcoh"],
        "Kosten_H2_€/100km": r["k_h2"],
        "Kosten_Diesel_€/100km": r["k_diesel"],
        "Kosten_BEV_€/100km": r["k_bev"],
        "CO2_H2_kg/100km": r["co2_h2"],
        "CO2_Diesel_kg/100km": r["co2_diesel"],
        "CO2_BEV_kg/100km": r["co2_bev"],
        "H2_Produktion_t/a": r["M_h2_jahr"]/1000.0,
    })
scan = pd.DataFrame(rows)
num_cols = [c for c in scan.columns if pd.api.types.is_numeric_dtype(scan[c])]

xcol = st.selectbox("X‑Achse", num_cols, index=num_cols.index("Strompreis_€/kWh") if "Strompreis_€/kWh" in num_cols else 0)
ycol = st.selectbox("Y‑Achse", num_cols, index=num_cols.index("LCOH_€/kg") if "LCOH_€/kg" in num_cols else 1)
cvar = st.selectbox("Farbkodierung", ["(keine)"] + num_cols, index=0)

try:
    rxy = scan[[xcol, ycol]].corr(method="pearson").iloc[0,1]
    title = f"{xcol} vs. {ycol} — r={rxy:.2f}"
except Exception:
    title = f"{xcol} vs. {ycol}"

fig_sc = px.scatter(scan, x=xcol, y=ycol, color=(None if cvar=="(keine)" else cvar), template="bosch",
                    hover_data=["Volllaststunden","Strompreis_€/kWh"], title=title)
fig_sc.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#999")))
fig_sc.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
st.plotly_chart(fig_sc, use_container_width=True)

if st.checkbox("Korrelationsmatrix anzeigen"):
    corrm = scan[num_cols].corr()
    st.plotly_chart(px.imshow(corrm, text_auto=True, aspect="auto", template="bosch", title="Korrelationsmatrix (Pearson)"
                              ).update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")),
                    use_container_width=True)

st.markdown("---")
st.caption("V2.6 — Header-Icons entfernt • Legenden transparent • H₂-Preis ab 1.50 €/kg.")
