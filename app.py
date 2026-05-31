import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from simulation import (
    BurnoutModel, SimConfig, run_monte_carlo,
    EXAM_PERIODS, STATE_COLORS, T_AT_RISK, T_BURNOUT,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi Burnout Mahasiswa — ABM Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1565c0;
        margin-bottom: 0.5rem;
    }
    .metric-card.danger  { border-left-color: #E53935; }
    .metric-card.warning { border-left-color: #FB8C00; }
    .metric-card.success { border-left-color: #43A047; }
    .metric-card.info    { border-left-color: #1E88E5; }
    .metric-val { font-size: 1.9rem; font-weight: 800; color: #1a237e; }
    .metric-lbl { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-sub { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

    /* Section headers */
    .section-header {
        font-size: 1.05rem; font-weight: 700;
        color: #1a237e; border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f8f9ff; }

    /* Info box */
    .info-box {
        background: #e8f5e9; border-left: 4px solid #43A047;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.85rem; margin-top: 0.5rem;
    }
    .warn-box {
        background: #fff3e0; border-left: 4px solid #FB8C00;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.85rem; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Parameter Panel
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Panel Kontrol Simulasi")
    st.markdown("---")

    # Mode
    mode = st.radio(
        "Mode Simulasi",
        ["🔬 Single Run (cepat)", "📊 Monte Carlo (statistik)"],
        help="Single Run: 1 simulasi deterministik. Monte Carlo: banyak iterasi untuk CI 95%."
    )
    is_mc = "Monte" in mode

    st.markdown("---")
    st.markdown("**⚙️ Parameter Model**")

    n_agents = st.slider("Jumlah Agen (Mahasiswa)", 20, 150, 80, 10,
                         help="Total populasi mahasiswa dalam simulasi")
    steps    = st.slider("Durasi Simulasi (hari)", 30, 150, 100, 10,
                         help="Panjang simulasi. 100 ≈ 1 semester")

    n_runs = 30  # default
    if is_mc:
        n_runs = st.slider("Iterasi Monte Carlo", 10, 100, 40, 10,
                           help="Lebih banyak = lebih akurat, tapi lebih lambat")

    st.markdown("---")
    st.markdown("**🤝 Intervensi**")

    ps_level  = st.slider("Peer Support Level (PS)", 0.0, 1.0, 0.0, 0.05,
                          help="Seberapa kuat dukungan teman sebaya. 0 = tidak ada.")
    ps_reactive = st.checkbox("PS Reaktif", value=False,
                              help="Jika aktif, PS hanya muncul saat ada tetangga yang burnout.")
    sc_level  = st.slider("Self-Care Level (SC)", 0.0, 1.0, 0.0, 0.05,
                          help="Intensitas kebiasaan self-care harian agen.")
    cbt_prog  = st.slider("Program CBT/Mindfulness", 0.0, 1.0, 0.0, 0.05,
                          help="Kekuatan program CBT. Meningkatkan skill koping agen.")

    st.markdown("---")
    st.markdown("**🌐 Jaringan Sosial**")

    net_type = st.selectbox("Tipe Jaringan", ["small_world", "scale_free"],
                            help="Small World: realistis (Watts-Strogatz). Scale Free: hub-dominated (Barabasi-Albert).")
    gamma    = st.slider("Laju Kontagion Burnout (γ)", 0.00, 0.40, 0.15, 0.01,
                         help="Seberapa mudah burnout menyebar antar teman. γ tinggi = lebih menular.")

    st.markdown("---")
    st.markdown("**📚 Stressor Akademik**")
    lambda_base = st.slider("Stressor Harian Dasar (λ)", 0.01, 0.30, 0.10, 0.01)
    lambda_peak = st.slider("Stressor Puncak Ujian (λ+)", 0.10, 1.00, 0.50, 0.05)

    st.markdown("---")
    seed = st.number_input("Random Seed", 0, 9999, 42, 1,
                           help="Ubah untuk mendapatkan populasi acak yang berbeda.")

    run_btn = st.button("▶ Jalankan Simulasi", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888; line-height:1.6'>
    📖 <b>Tugas Besar</b> Pemodelan & Simulasi<br>
    🧠 Topik: Burnout Mahasiswa (ABM)<br>
    📅 Minggu 16 — UAS Evaluasi II<br>
    🔧 burnout_abm_v3.py (Monte Carlo)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 Simulasi Penyebaran Burnout Mahasiswa — ABM Dashboard</h1>
  <p>Evaluasi Efektivitas Intervensi Peer Support & Self-Care menggunakan Agent-Based Modeling</p>
</div>
""", unsafe_allow_html=True)

# ── Tab Layout ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Hasil Simulasi",
    "🗺️ Jaringan Sosial",
    "🔬 Analitik Lanjut",
    "⚖️ Komparasi Skenario",
    "📖 Tentang Model",
])


# ══════════════════════════════════════════════════════════════════
# SIMULATION CACHE
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cached_single(n_agents, steps, ps_level, sc_level, cbt_prog,
                  ps_reactive, net_type, gamma, lambda_base, lambda_peak, seed):
    cfg = SimConfig(
        num_agents=n_agents, steps=steps,
        ps_level=ps_level, sc_level=sc_level, cbt_program=cbt_prog,
        ps_reactive=ps_reactive, network_type=net_type,
        gamma=gamma, lambda_base=lambda_base, lambda_peak=lambda_peak,
        seed=seed,
    )
    model = BurnoutModel(cfg)
    model.run_all()
    return model


@st.cache_data(show_spinner=False)
def cached_mc(n_agents, steps, n_runs, ps_level, sc_level, cbt_prog,
              ps_reactive, net_type, gamma, lambda_base, lambda_peak):
    cfg = SimConfig(
        num_agents=n_agents, steps=steps,
        ps_level=ps_level, sc_level=sc_level, cbt_program=cbt_prog,
        ps_reactive=ps_reactive, network_type=net_type,
        gamma=gamma, lambda_base=lambda_base, lambda_peak=lambda_peak,
    )
    return run_monte_carlo(cfg, n_runs)


# ══════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ══════════════════════════════════════════════════════════════════
if "ran" not in st.session_state:
    st.session_state.ran = False
if "last_mode" not in st.session_state:
    st.session_state.last_mode = is_mc
if st.session_state.last_mode != is_mc:
    # Mode changed — force re-run
    st.session_state.ran = False
    st.session_state.last_mode = is_mc

params_single = (n_agents, steps, ps_level, sc_level, cbt_prog,
                 ps_reactive, net_type, gamma, lambda_base, lambda_peak, seed)
params_mc     = (n_agents, steps, n_runs if is_mc else 30,
                 ps_level, sc_level, cbt_prog,
                 ps_reactive, net_type, gamma, lambda_base, lambda_peak)

if run_btn or not st.session_state.ran:
    st.session_state.ran = True
    with st.spinner("⏳ Menjalankan simulasi..."):
        st.session_state.model = cached_single(*params_single)
        if is_mc:
            st.session_state.mc = cached_mc(*params_mc)
        else:
            st.session_state.mc = None

if "model" not in st.session_state:
    st.warning("⬅️ Klik **▶ Jalankan Simulasi** di sidebar untuk memulai.")
    st.stop()

model = st.session_state.model
mc    = st.session_state.mc

df   = model.get_records_df()
t    = np.arange(len(df))
EXAM = [e for e in EXAM_PERIODS if e <= steps]


# ══════════════════════════════════════════════════════════════════
# TAB 1 — HASIL SIMULASI
# ══════════════════════════════════════════════════════════════════
with tab1:
    # ── KPI Cards ────────────────────────────────────────────────
    final = df.iloc[-1]
    n = n_agents
    pct_b = final["n_burnout"]  / n * 100
    pct_n = final["n_normal"]   / n * 100
    pct_r = final["n_recovery"] / n * 100
    avg_b = final["avg_burnout"]
    avg_r = final["avg_resilience"]
    rt = model.get_recovery_times()
    med_rt = float(np.median(rt)) if rt else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f"{avg_b:.3f}", "Burnout Level Akhir (B̄)", f"Ambang burnout: {T_BURNOUT}", "danger" if avg_b >= T_BURNOUT else "warning" if avg_b >= T_AT_RISK else "success"),
        (c2, f"{pct_b:.1f}%", "Populasi Burnout", f"{int(final['n_burnout'])} dari {n} agen", "danger" if pct_b > 50 else "warning"),
        (c3, f"{pct_n:.1f}%", "Populasi Normal", f"{int(final['n_normal'])} agen sehat", "success"),
        (c4, f"{avg_r:.3f}", "Resilience Rata-rata", "Semakin tinggi = lebih tahan burnout", "info"),
        (c5, f"{med_rt:.0f} hr", "Median Waktu Pulih", "Burnout → Normal (hari)", "info"),
    ]
    for col, val, lbl, sub, cls in cards:
        col.markdown(f"""
        <div class="metric-card {cls}">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main trajectory chart ────────────────────────────────────
    st.markdown('<div class="section-header">📈 Trajektori Burnout Level B̄ sepanjang Simulasi</div>', unsafe_allow_html=True)

    fig = go.Figure()

    if is_mc and mc:
        m, lo, hi = mc["ab"]
        t_mc = np.arange(len(m))
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_mc, t_mc[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor="rgba(21,101,192,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="CI 95%",
        ))
        fig.add_trace(go.Scatter(x=t_mc, y=m, name=f"Mean (MC {mc['n_runs']} iter)",
                                 line=dict(color="#1565c0", width=2.5)))
    else:
        fig.add_trace(go.Scatter(x=t, y=df["avg_burnout"], name="Burnout Level B̄",
                                 line=dict(color="#1565c0", width=2.5)))

    # Ambang
    t_full = np.arange(steps + 1)
    fig.add_hline(y=T_AT_RISK, line_dash="dash", line_color="#FB8C00",
                  annotation_text="At-Risk (0.30)", annotation_position="right")
    fig.add_hline(y=T_BURNOUT, line_dash="dash", line_color="#E53935",
                  annotation_text="Burnout (0.60)", annotation_position="right")

    # Exam periods
    for ep_start, ep_end in [(EXAM[0], EXAM[9]) if len(EXAM)>=10 else (0,0),
                              (EXAM[10], EXAM[19]) if len(EXAM)>=20 else (0,0)]:
        if ep_start:
            fig.add_vrect(x0=ep_start, x1=ep_end, fillcolor="red",
                          opacity=0.07, line_width=0, annotation_text="Ujian",
                          annotation_position="top left")

    fig.update_layout(
        xaxis_title="Langkah Simulasi (hari)", yaxis_title="Rata-rata Burnout Level",
        yaxis=dict(range=[0, 1.05]),
        height=380, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=80, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── State population chart ───────────────────────────────────
    st.markdown('<div class="section-header">👥 Komposisi Populasi per State</div>', unsafe_allow_html=True)

    if is_mc and mc:
        m_pb, lo_pb, hi_pb = mc["pb"]
        m_pn, lo_pn, hi_pn = mc["pn"]
        t_mc = np.arange(len(m_pb))
        fig2 = go.Figure()
        for m_, lo_, hi_, name, color in [
            (m_pn, lo_pn, hi_pn, "Normal",  "#43A047"),
            (m_pb, lo_pb, hi_pb, "Burnout", "#E53935"),
        ]:
            fig2.add_trace(go.Scatter(
                x=np.concatenate([t_mc, t_mc[::-1]]),
                y=np.concatenate([hi_, lo_[::-1]]),
                fill="toself", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.15,)}",
                line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
            ))
            fig2.add_trace(go.Scatter(x=t_mc, y=m_, name=f"% {name}",
                                      line=dict(color=color, width=2.5)))
    else:
        fig2 = go.Figure()
        for key, label, color in [
            ("n_normal","Normal","#43A047"), ("n_atrisk","At-Risk","#FB8C00"),
            ("n_recovery","Recovery","#1E88E5"), ("n_burnout","Burnout","#E53935"),
        ]:
            fig2.add_trace(go.Scatter(
                x=t, y=df[key] / n * 100,
                name=f"% {label}", stackgroup="one",
                line=dict(color=color), fillcolor=color,
            ))

    for ep_start, ep_end in [(EXAM[0], EXAM[9]) if len(EXAM)>=10 else (0,0),
                              (EXAM[10], EXAM[19]) if len(EXAM)>=20 else (0,0)]:
        if ep_start:
            fig2.add_vrect(x0=ep_start, x1=ep_end, fillcolor="red", opacity=0.07, line_width=0)

    fig2.update_layout(
        xaxis_title="Hari", yaxis_title="% Populasi",
        yaxis=dict(range=[0, 105]),
        height=340, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=80, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stressor timeline ────────────────────────────────────────
    with st.expander("📊 Lihat detail: Stressor & Resilience Timeline"):
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Stressor Harian", "Resilience Rata-rata"])
        fig3.add_trace(go.Bar(x=t, y=df["stressor"], name="Stressor",
                              marker_color="#E53935", opacity=0.7), row=1, col=1)
        fig3.add_trace(go.Scatter(x=t, y=df["avg_resilience"], name="Resilience",
                                  line=dict(color="#43A047", width=2)), row=1, col=2)
        for ep_s, ep_e in [(EXAM[0], EXAM[9]) if len(EXAM)>=10 else (0,0),
                            (EXAM[10], EXAM[19]) if len(EXAM)>=20 else (0,0)]:
            if ep_s:
                for col in [1, 2]:
                    fig3.add_vrect(x0=ep_s, x1=ep_e, fillcolor="red", opacity=0.07, line_width=0, row=1, col=col)
        fig3.update_layout(height=280, template="plotly_white", showlegend=False,
                           margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — JARINGAN SOSIAL
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🗺️ Visualisasi Jaringan Sosial Antar Agen</div>', unsafe_allow_html=True)

    col_info, col_net = st.columns([1, 3])

    with col_info:
        states = model.get_agent_states()
        state_counts = {s: states.count(s) for s in ["Normal","At-Risk","Burnout","Recovery"]}
        for s, cnt in state_counts.items():
            pct = cnt / n_agents * 100
            color = STATE_COLORS[s]
            st.markdown(f"""
            <div style="background:{color}20; border-left:4px solid {color};
                        padding:0.5rem 0.8rem; border-radius:6px; margin-bottom:0.5rem;">
              <b style="color:{color}">{s}</b><br>
              <span style="font-size:1.4rem;font-weight:800">{cnt}</span>
              <span style="color:#666"> agen ({pct:.1f}%)</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        G = model.G
        st.metric("Jumlah Node", G.number_of_nodes())
        st.metric("Jumlah Edge", G.number_of_edges())
        st.metric("Avg Degree", f"{sum(d for _,d in G.degree())/G.number_of_nodes():.1f}")
        try:
            import networkx as nx
            st.metric("Clustering Coef.", f"{nx.average_clustering(G):.3f}")
        except:
            pass

    with col_net:
        import networkx as nx
        G = model.G
        states = model.get_agent_states()
        # burnouts dict built per-state below

        # Layout
        if n_agents <= 60:
            pos = nx.spring_layout(G, seed=42, k=1.5/math.sqrt(n_agents))
        else:
            pos = nx.kamada_kawai_layout(G)

        # Build plotly network
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0,y0 = pos[u]; x1,y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#cccccc"), hoverinfo="none",
        ))

        # Build per-node lookup keyed by node ID (not enumerate index)
        node_ids = list(G.nodes())
        node_state   = {uid: model.agents[uid].state   for uid in node_ids}
        node_burnout = {uid: model.agents[uid].burnout for uid in node_ids}

        for state in ["Normal","At-Risk","Burnout","Recovery"]:
            idxs = [uid for uid in node_ids if node_state[uid] == state]
            if not idxs: continue
            xs = [pos[uid][0] for uid in idxs]
            ys = [pos[uid][1] for uid in idxs]
            bs_val = [node_burnout[uid] for uid in idxs]
            fig_net.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name=state,
                marker=dict(
                    size=[6 + b*14 for b in bs_val],
                    color=STATE_COLORS[state],
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=[f"Agen {uid}<br>State: {state}<br>B={b:.3f}"
                      for uid, b in zip(idxs, bs_val)],
                hovertemplate="%{text}<extra></extra>",
            ))

        fig_net.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500, template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption("Ukuran node ∝ burnout level. Warna = state agen di akhir simulasi.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALITIK LANJUT
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔬 Analitik Distribusi & Waktu Pemulihan</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Histogram burnout akhir
        final_b = model.get_final_burnouts()
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=final_b, nbinsx=25,
            marker_color="#1565c0", opacity=0.75,
            name="Distribusi B akhir",
        ))
        fig_h.add_vline(x=T_AT_RISK, line_dash="dash", line_color="#FB8C00", annotation_text="At-Risk")
        fig_h.add_vline(x=T_BURNOUT, line_dash="dash", line_color="#E53935", annotation_text="Burnout")
        fig_h.update_layout(
            title="Distribusi Burnout Level Akhir",
            xaxis_title="Burnout Level B", yaxis_title="Jumlah Agen",
            height=320, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        # Recovery time distribution
        rt = model.get_recovery_times()
        if rt:
            fig_rt = go.Figure()
            fig_rt.add_trace(go.Histogram(
                x=rt, nbinsx=20,
                marker_color="#43A047", opacity=0.75,
                name="Waktu Pemulihan",
            ))
            fig_rt.add_vline(x=np.median(rt), line_dash="dash", line_color="#1565c0",
                             annotation_text=f"Median={np.median(rt):.1f}")
            fig_rt.update_layout(
                title="Distribusi Waktu Pemulihan (Burnout → Normal)",
                xaxis_title="Hari", yaxis_title="Frekuensi",
                height=320, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0),
            )
            st.plotly_chart(fig_rt, use_container_width=True)
        else:
            st.info("Tidak ada agen yang mengalami siklus pemulihan penuh pada konfigurasi ini.")

    # Scatter: Vulnerability vs Burnout akhir
    st.markdown("---")
    st.markdown('<div class="section-header">🔎 Profil Agen: Vulnerability vs Burnout Akhir</div>', unsafe_allow_html=True)

    agent_data = pd.DataFrame({
        "Agen"        : range(n_agents),
        "Vulnerability": [model.agents[i].vulnerability for i in range(n_agents)],
        "Resilience"  : [model.agents[i].resilience     for i in range(n_agents)],
        "Burnout Akhir": [model.agents[i].burnout        for i in range(n_agents)],
        "State"       : [model.agents[i].state           for i in range(n_agents)],
        "Help Seeking": [model.agents[i].help_seeking    for i in range(n_agents)],
    })

    fig_sc = px.scatter(
        agent_data, x="Vulnerability", y="Burnout Akhir",
        color="State", size="Resilience",
        color_discrete_map=STATE_COLORS,
        hover_data=["Agen","Help Seeking"],
        title="Vulnerability vs Burnout Level Akhir (ukuran ∝ Resilience)",
    )
    fig_sc.add_hline(y=T_AT_RISK, line_dash="dash", line_color="#FB8C00")
    fig_sc.add_hline(y=T_BURNOUT, line_dash="dash", line_color="#E53935")
    fig_sc.update_layout(height=380, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Agent table
    with st.expander("📋 Lihat Tabel Data Agen"):
        st.dataframe(
            agent_data.sort_values("Burnout Akhir", ascending=False)
                      .style.background_gradient(subset=["Burnout Akhir"], cmap="RdYlGn_r"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — KOMPARASI SKENARIO
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⚖️ Perbandingan 5 Skenario What-If (Monte Carlo)</div>', unsafe_allow_html=True)

    st.info("Tab ini selalu menjalankan 5 skenario preset dengan 40 iterasi MC. Hasilnya independen dari slider di sidebar.", icon="ℹ️")

    PRESET_SCENARIOS = [
        ("S1: Tanpa Intervensi",          dict(ps_level=0.0, sc_level=0.0, cbt_program=0.0, ps_reactive=False, gamma=0.15)),
        ("S2: Peer Support Reaktif",      dict(ps_level=0.8, sc_level=0.0, cbt_program=0.0, ps_reactive=True,  gamma=0.15)),
        ("S3: Self-Care + CBT Preventif", dict(ps_level=0.0, sc_level=0.6, cbt_program=0.6, ps_reactive=False, gamma=0.15)),
        ("S4: Kombinasi Lengkap",         dict(ps_level=0.7, sc_level=0.6, cbt_program=0.5, ps_reactive=True,  gamma=0.15)),
        ("S5: Kerentanan Tinggi (γ=0.22)",dict(ps_level=0.6, sc_level=0.6, cbt_program=0.7, ps_reactive=False, gamma=0.22)),
    ]
    PRESET_COLORS = ["#E53935","#FB8C00","#43A047","#1E88E5","#8E24AA"]

    @st.cache_data(show_spinner=False)
    def run_all_presets(n_agents, steps):
        results = []
        for name, kw in PRESET_SCENARIOS:
            cfg = SimConfig(num_agents=n_agents, steps=steps, **kw)
            mc  = run_monte_carlo(cfg, n_runs=40)
            results.append({"name": name, **mc})
        return results

    with st.spinner("⏳ Menjalankan 5 skenario preset..."):
        presets = run_all_presets(n_agents, steps)

    t_p = np.arange(steps + 1)

    # Ribbon burnout comparison
    fig_cmp = go.Figure()
    for res, color in zip(presets, PRESET_COLORS):
        m, lo, hi = res["ab"]
        fig_cmp.add_trace(go.Scatter(
            x=np.concatenate([t_p, t_p[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.12,)}",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))
        fig_cmp.add_trace(go.Scatter(
            x=t_p, y=m, name=res["name"],
            line=dict(color=color, width=2.2),
        ))

    for ep_s, ep_e in [(EXAM[0], EXAM[9]) if len(EXAM)>=10 else (0,0),
                        (EXAM[10], EXAM[19]) if len(EXAM)>=20 else (0,0)]:
        if ep_s:
            fig_cmp.add_vrect(x0=ep_s, x1=ep_e, fillcolor="red", opacity=0.06, line_width=0)
    fig_cmp.add_hline(y=T_AT_RISK, line_dash="dash", line_color="#FB8C00", annotation_text="At-Risk")
    fig_cmp.add_hline(y=T_BURNOUT, line_dash="dash", line_color="#E53935", annotation_text="Burnout")
    fig_cmp.update_layout(
        title="Perbandingan Rata-rata Burnout Level (mean ± CI 95%)",
        xaxis_title="Hari", yaxis_title="B̄",
        yaxis=dict(range=[0,1.05]),
        height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=80, t=40, b=0),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Summary bar chart
    st.markdown("---")
    col_bar1, col_bar2 = st.columns(2)

    b_finals = [res["final_b"].mean() for res in presets]
    pct_burns = [res["pb"][0][-1] for res in presets]
    pct_norms = [res["pn"][0][-1] for res in presets]
    med_rts   = [float(np.median(res["rec_t"])) if res["rec_t"] else 0 for res in presets]
    names_short = [r["name"].split(":")[0] for r in presets]

    with col_bar1:
        fig_bar = go.Figure(go.Bar(
            x=names_short, y=b_finals,
            marker_color=PRESET_COLORS, opacity=0.85,
            text=[f"{v:.3f}" for v in b_finals], textposition="outside",
        ))
        fig_bar.add_hline(y=T_BURNOUT, line_dash="dash", line_color="#E53935")
        fig_bar.update_layout(title="Rata-rata Burnout Level Akhir", height=300,
                              template="plotly_white", yaxis=dict(range=[0,1.15]),
                              margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_bar2:
        fig_bar2 = go.Figure()
        fig_bar2.add_trace(go.Bar(x=names_short, y=pct_burns, name="% Burnout",
                                  marker_color=PRESET_COLORS, opacity=0.7,
                                  text=[f"{v:.1f}%" for v in pct_burns], textposition="outside"))
        fig_bar2.add_trace(go.Bar(x=names_short, y=pct_norms, name="% Normal",
                                  marker_color=["#43A047"]*5, opacity=0.5,
                                  text=[f"{v:.1f}%" for v in pct_norms], textposition="outside"))
        fig_bar2.update_layout(title="% Populasi Burnout vs Normal Akhir",
                               barmode="group", height=300, template="plotly_white",
                               yaxis=dict(range=[0,125]),
                               margin=dict(l=0,r=0,t=40,b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_bar2, use_container_width=True)

    # Summary table
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Tabel Ringkasan Komparasi</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Skenario"          : [r["name"] for r in presets],
        "B̄ Akhir (mean)"   : [f"{v:.3f}" for v in b_finals],
        "% Burnout Akhir"   : [f"{v:.1f}%" for v in pct_burns],
        "% Normal Akhir"    : [f"{v:.1f}%" for v in pct_norms],
        "Median Rec. (hari)": [f"{v:.1f}" if v > 0 else "N/A" for v in med_rts],
        "Rekomendasi"       : [
            "❌ Tidak direkomendasikan",
            "⚡ Untuk kondisi darurat",
            "✅ Terbaik untuk proteksi jangka panjang",
            "✅ Seimbang — direkomendasikan",
            "⚠️ Perlu perhatian ekstra",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
