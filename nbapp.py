import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pulp
import os
import math
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression

# 1. SAYFA KONFİGÜRASYONU
st.set_page_config(page_title="NBA Analytics", layout="wide", page_icon="🏀")

# --- YARDIMCI FONKSİYONLAR ---
def fmt_money(val):
    if pd.isna(val) or val == 0:
        return "$0.00 M"
    return f"${val/1000000:.2f} M"

def format_p(p_val):
    if p_val < 0.0001:
        return "< 0.0001"
    return f"{p_val:.4f}"

# 2. VERİ YÜKLEME VE TEMİZLEME
@st.cache_data
def load_data():
    file_path = "nba_gercek_maasli_veri.csv"
    if not os.path.exists(file_path):
        st.error(f"Kritik Hata: '{file_path}' bulunamadı. Lütfen önce 'veri_birlestir.py' çalıştırın.")
        st.stop()
    df = pd.read_csv(file_path)
    
    cols_to_fix = ["WS", "Salary", "Current_Salary", "PTS", "AST", "TRB", "STL", "BLK", "PER", "3PA", "TS%", "Age", "USG%", "BPM", "VORP", "G"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if "USG%" not in df.columns: df["USG%"] = 20.0
    if "BPM" not in df.columns: df["BPM"] = 0.0
    if "VORP" not in df.columns: df["VORP"] = 0.0
    if "G" not in df.columns: df["G"] = 70
    
    df['Season'] = df['Season'].astype(str)
    return df

df = load_data()

latest_season = df['Season'].max()
active_players = df[df['Season'] == latest_season]['Player'].unique()
latest_salaries = df.groupby('Player')['Current_Salary'].first().to_dict()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
for fallback in ['Pos', 'pos', 'Pos_per', 'pos_per', 'Pos_adv', 'Tm', 'Season']:
    if fallback in df.columns and fallback not in cat_cols:
        cat_cols.append(fallback)
cat_cols = list(set(cat_cols))

pos_col = next((c for c in ['Pos', 'pos', 'Pos_per', 'pos_per', 'Pos_adv'] if c in df.columns), None)

# 3. YAN PANEL (SIDEBAR) & OYUNCU LİSTESİ HAZIRLIĞI
st.sidebar.title("🏀 NBA Analytics Dashboard")
st.sidebar.markdown("---")

include_retired = st.sidebar.toggle("Emekli Oyuncuları Göster", value=True)
if include_retired:
    available_players = sorted(df["Player"].dropna().unique())
    f_df = df.copy()
else:
    available_players = sorted([p for p in df["Player"].unique() if p in active_players])
    f_df = df[df["Player"].isin(active_players)].copy()

if 'roster_selection' not in st.session_state:
    st.session_state['roster_selection'] = available_players[:5] if len(available_players) >= 5 else available_players

menu = st.sidebar.radio("Analiz Modülleri:", [
    "1. 👤 Oyuncu Profili & Radar Analizi",
    "2. 🏟️ Takım Kıyaslama & Benchmarking",
    "3. 📈 Piyasa Verimlilik Sınırı (Moneyball)",
    "4. 🚑 Sakatlık & Risk Senaryo Analizi",
    "5. 💎 Fiyat/Performans Keşfi (Gems)",
    "6. ⏳ Dönemsel Evrim Analizi (2010-2025)",
    "7. 🔬 Akademik İstatistik & Tahmin Modelleri"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;">
        <span style="color: #808495; font-size: 12px;">PROJE ÜYELERİ</span><br>
        <b style="color: white;">Batuhan Yeniyurt</b><br>
        <b style="color: white;">Haydarhan Kirazlı</b><br>
        <b style="color: white;">Osman Oskay</b>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODÜLLER 1-6
# ---------------------------------------------------------
if menu == "1. 👤 Oyuncu Profili & Radar Analizi":
    st.header("👤 Oyuncu Yetenek ve Değer Analizi")
    c1, c2 = st.columns(2)
    with c1: p1 = st.selectbox("Ana Oyuncuyu Seçin", available_players, index=0)
    with c2: p2 = st.selectbox("Karşılaştırılacak Oyuncu", ["Seçilmedi"] + list(available_players))

    p1_data = df[df["Player"] == p1].sort_values("Season").iloc[-1:]
    p1_guncel_maas = latest_salaries.get(p1, 0)
    p1_normal_maas = p1_data["Salary"].values[0] if "Salary" in p1_data else 0
    
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win Shares (WS)", round(p1_data["WS"].values[0], 2))
    m2.metric("O Sezonki Normal Maaşı", fmt_money(p1_normal_maas), help="Oyuncunun seçilen spesifik sezonda takımdan aldığı maaştır.")
    
    if p2 != "Seçilmedi":
        p2_data = df[df["Player"] == p2].sort_values("Season").iloc[-1:]
        p2_guncel_maas = latest_salaries.get(p2, 0)
        p2_normal_maas = p2_data["Salary"].values[0] if "Salary" in p2_data else 0
        m3.metric(f"{p2} WS", round(p2_data["WS"].values[0], 2), delta=round(p2_data["WS"].values[0] - p1_data["WS"].values[0], 2))
        m4.metric(f"{p2} Sezonluk Maaş", fmt_money(p2_normal_maas), delta=fmt_money(p2_normal_maas - p1_normal_maas), delta_color="inverse")
    else:
        m3.metric("PER (Verimlilik)", round(p1_data["PER"].values[0], 1))
        m4.metric("Güncel Maaşı (2024-25)", fmt_money(p1_guncel_maas), help="Oyuncunun günümüzde ligden aldığı son kontrat maaşıdır.")

    radar_cols = ["PTS", "AST", "TRB", "STL", "BLK", "WS"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[p1_data[m].values[0] for m in radar_cols], theta=radar_cols, fill='toself', name=p1, line_color='#00FF00'))
    if p2 != "Seçilmedi":
        fig.add_trace(go.Scatterpolar(r=[p2_data[m].values[0] for m in radar_cols], theta=radar_cols, fill='toself', name=p2, line_color='#FF4B4B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Yetenek Radarı Karşılaştırması")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"📊 {p1} Tarihsel İstatistikleri")
    disp_df = df[df["Player"] == p1].sort_values("Season", ascending=False).copy()
    if "Salary" in disp_df.columns: disp_df["Salary"] = disp_df["Salary"].apply(fmt_money)
    if "Current_Salary" in disp_df.columns: disp_df["Current_Salary"] = disp_df["Current_Salary"].apply(fmt_money)
    st.dataframe(disp_df)

elif menu == "2. 🏟️ Takım Kıyaslama & Benchmarking":
    st.header("🏟️ Gelişmiş Takım Tarihsel Kıyaslaması")
    st.write("Takımların tarihsel performanslarını sadece birbirleriyle değil, lig ortalamasıyla da kıyaslayın.")
    
    teams = sorted(df["Tm"].dropna().unique())
    col1, col2, col3, col4 = st.columns(4)
    t1 = col1.selectbox("1. Takım", teams, index=0)
    t2 = col2.selectbox("2. Takım", teams, index=1 if len(teams)>1 else 0)
    
    adv_metrics = [c for c in ["WS", "PTS", "Salary", "3PA", "AST", "TRB", "BPM", "VORP", "Age"] if c in df.columns]
    m = col3.selectbox("Karşılaştırılacak Metrik", adv_metrics)
    
    show_league_avg = col4.checkbox("Lig Ortalamasını Göster", value=True, help="Grafiğe tüm NBA'in o sezonki ortalamasını ekler.")
    
    agg_func = 'mean' if m in ['Age', 'BPM'] else 'sum'
    t_data = df[df["Tm"].isin([t1, t2])].groupby(["Season", "Tm"])[m].agg(agg_func).reset_index()
    
    if show_league_avg:
        league_data = df.groupby(["Season", "Tm"])[m].agg(agg_func).reset_index()
        league_avg = league_data.groupby("Season")[m].mean().reset_index()
        league_avg["Tm"] = "🏀 NBA ORTALAMASI"
        t_data = pd.concat([t_data, league_avg], ignore_index=True)
    
    fig = px.line(t_data, x="Season", y=m, color="Tm", markers=True, title=f"Yıllara Göre Takım {m} Karşılaştırması")
    if show_league_avg:
        for trace in fig.data:
            if trace.name == "🏀 NBA ORTALAMASI":
                trace.line.dash = 'dash'
                trace.line.width = 4
                trace.line.color = 'white'
                
    st.plotly_chart(fig, use_container_width=True)

elif menu == "3. 📈 Piyasa Verimlilik Sınırı (Moneyball)":
    st.header("📈 Dinamik Piyasa Verimlilik Sınırı")
    st.success("💰 Kendi Regresyon Modelinizi Kurun: Sadece Maaş vs WS değil, X ve Y eksenlerini kendiniz belirleyerek gizli değerleri (Gems) keşfedin.")
    
    c1, c2, c3, c4 = st.columns(4)
    sel_year = c1.selectbox("Sezon Seç", sorted(df["Season"].unique(), reverse=True))
    x_col = c2.selectbox("Maliyet/Kriter (X Ekseni)", [c for c in ["Salary", "USG%", "Age", "3PA"] if c in df.columns])
    y_col = c3.selectbox("Performans (Y Ekseni)", [c for c in ["WS", "BPM", "VORP", "PER", "PTS"] if c in df.columns])
    min_g = c4.slider("Minimum Maç (G) Filtresi", 1, 82, 30, help="Sakatlıktan az oynamış oyuncuların analizi bozmasını engeller.")
    
    year_df = f_df[(f_df["Season"] == sel_year) & (f_df[x_col] > 0) & (f_df["G"] >= min_g)].copy()
    
    if len(year_df) > 5:
        X = year_df[[x_col]].values
        y = year_df[y_col].values
        reg = LinearRegression().fit(X, y)
        year_df["PiyasaTrendi"] = reg.predict(X)
        year_df["Verimlilik (Fark)"] = year_df[y_col] - year_df["PiyasaTrendi"]
        
        hover_data = {x_col: True, y_col: True, "Verimlilik (Fark)": ':.2f'}
        if x_col == "Salary":
            year_df["Formatlı_Maaş"] = year_df["Salary"].apply(fmt_money)
            hover_data["Salary"] = False
            hover_data["Formatlı_Maaş"] = True
            
        fig = px.scatter(year_df, x=x_col, y=y_col, hover_name="Player", hover_data=hover_data, color="Verimlilik (Fark)", color_continuous_scale="RdYlGn", size="PTS", title=f"{sel_year} Sezonu {x_col} vs {y_col} Dağılımı")
        fig.add_traces(go.Scatter(x=year_df[x_col], y=year_df["PiyasaTrendi"], mode='lines', name='Piyasa Ortalaması (Beklenti)', line=dict(color='white')))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Model Sonuçları: Piyasadaki Aşırılıklar")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.success(f"🟢 **Beklentiyi En Çok Aşanlar (Undervalued)**")
            best_df = year_df.sort_values("Verimlilik (Fark)", ascending=False).head(5)[["Player", x_col, y_col, "Verimlilik (Fark)"]]
            if x_col == "Salary": best_df[x_col] = best_df[x_col].apply(fmt_money)
            st.dataframe(best_df, hide_index=True)
            
        with col_t2:
            st.error(f"🔴 **Beklentinin En Çok Altında Kalanlar (Overvalued)**")
            worst_df = year_df.sort_values("Verimlilik (Fark)", ascending=True).head(5)[["Player", x_col, y_col, "Verimlilik (Fark)"]]
            if x_col == "Salary": worst_df[x_col] = worst_df[x_col].apply(fmt_money)
            st.dataframe(worst_df, hide_index=True)
    else:
        st.warning("Verimlilik analizi için bu seçenekte yeterli veri seti bulunamadı.")

elif menu == "4. 🚑 Sakatlık & Risk Senaryo Analizi":
    st.header("🚑 Sakatlık ve Risk Yönetimi")
    inj_player = st.selectbox("Sakatlanan Yıldız Oyuncu", available_players)
    missed_g = st.slider("Kaçırılacak Maç Sayısı", 1, 82, 20)
    bench_quality = st.select_slider("Yedek (Bench) Kalitesi", options=["Düşük", "Normal", "Yüksek"], value="Normal")
    bench_map = {"Düşük": 0.1, "Normal": 0.3, "Yüksek": 0.6}
    p_ws = df[df["Player"] == inj_player]["WS"].mean()
    net_loss = (p_ws * (missed_g / 82)) * (1 - bench_map[bench_quality])
    s1, s2, s3 = st.columns(3)
    s1.metric("Kayıp Galibiyet Beklentisi", f"-{round(net_loss, 2)}")
    s2.metric("Yeni Tahmini Galibiyet", round(50 - net_loss, 1))
    s3.metric("Risk Seviyesi", "Kritik 🚨" if net_loss > 4 else "Düşük ✅")
    sims = np.random.normal(50 - net_loss, 4, 1000)
    fig = px.histogram(sims, title="1.000 Senaryoda Olası Sezon Sonu Galibiyet Sayısı", color_discrete_sequence=['red'])
    fig.add_vline(x=42, line_dash="dash", line_color="white", annotation_text="Play-off Sınırı")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "5. 💎 Fiyat/Performans Keşfi (Gems)":
    st.header("💎 Moneyball Kelepirleri")
    max_sal = st.slider("Maksimum Maaş Sınırı (Milyon $)", 1, 15, 5) * 1000000
    gems = f_df[(f_df["Season"] == latest_season) & (f_df["Current_Salary"] <= max_sal) & (f_df["Current_Salary"] > 0)].copy()
    if not gems.empty:
        gems["ROI (WS/M$)"] = (gems["WS"] / (gems["Current_Salary"] / 1000000))
        gems = gems.sort_values("ROI (WS/M$)", ascending=False).head(15)
        st.plotly_chart(px.bar(gems, x="Player", y="ROI (WS/M$)", color="WS", color_continuous_scale="Greens"), use_container_width=True)
        gems["Formatlı Maaş"] = gems["Current_Salary"].apply(fmt_money)
        cols = ["Player", "Formatlı Maaş", "WS", "ROI (WS/M$)"]
        if "Age" in gems.columns: cols.append("Age")
        st.table(gems[cols])

elif menu == "6. ⏳ Dönemsel Evrim Analizi (2010-2025)":
    st.header("⏳ NBA Oyun Karakteristiğinin Evrimi")
    st.write("Basketbolun son 10 yılda nasıl bir dönüşüm geçirdiğini makro (Lig) ve mikro (Pozisyon) bazda inceleyin.")
    
    analysis_type = st.radio("İnceleme Perspektifi:", ["Lig Geneli Toplu Trend", "Pozisyona Göre Evrim Kırılımı"], horizontal=True)
    
    if analysis_type == "Lig Geneli Toplu Trend":
        metric = st.selectbox("İncelemek İstediğiniz Değişim", num_cols, index=num_cols.index("3PA") if "3PA" in num_cols else 0)
        evol = df.groupby("Season")[metric].mean().reset_index().sort_values("Season")
        fig = px.area(evol, x="Season", y=metric, markers=True, color_discrete_sequence=['#2ecc71'], title=f"Yıllara Göre Lig Geneli Ortalama {metric}")
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
        v2010 = evol.iloc[0][metric]
        v2025 = evol.iloc[-1][metric]
        pct = ((v2025 - v2010) / v2010) * 100 if v2010 != 0 else 0
        
        # --- CURRENT_SALARY İÇİN DÜZELTİLMİŞ PARA FORMATI KOD BLOĞU ---
        display_val = fmt_money(v2025) if metric in ["Salary", "Current_Salary"] else f"{v2025:.2f}"
        
        st.metric(label=f"Ortalama {metric} Değişimi (2010 → 2025)", value=display_val, delta=f"%{pct:.1f}")
    else:
        if pos_col:
            metric_pos = st.selectbox("Pozisyonlara Göre İncelenecek Metrik", num_cols, index=num_cols.index("3PA") if "3PA" in num_cols else 0)
            df_pos_clean = df.dropna(subset=[pos_col, metric_pos]).copy()
            df_pos_clean['P_Group'] = df_pos_clean[pos_col].astype(str).str.upper().str.extract(r'(PG|SG|SF|PF|C)')[0]
            df_pos_clean = df_pos_clean.dropna(subset=['P_Group'])
            evol_pos = df_pos_clean.groupby(["Season", "P_Group"])[metric_pos].mean().reset_index().sort_values("Season")
            fig2 = px.line(evol_pos, x="Season", y=metric_pos, color="P_Group", markers=True, title=f"Pozisyonların {metric_pos} Evrimi")
            fig2.update_xaxes(type='category')
            st.plotly_chart(fig2, use_container_width=True)
            st.info("💡 **Analiz Notu:** Çizgilerin birbirine yaklaşması 'Pozisyonsuz Basketbol' (Positionless Basketball) devrimini, ayrışması ise rollerin keskinleştiğini gösterir.")
        else:
            st.error("Veri setinizde pozisyon sütunu bulunamadı.")

# ---------------------------------------------------------
# MODÜL 7: AKADEMİK İSTATİSTİK & TAHMİN MODELLERİ
# ---------------------------------------------------------
elif menu == "7. 🔬 Akademik İstatistik & Tahmin Modelleri":
    st.header("🔬 İstatistik Laboratuvarı ve Gelişmiş AI")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Hipotez Testleri", 
        "📊 Çoklu Regresyon", 
        "🤖 K-Means", 
        "🎲 Monte Carlo & AI Kadro Simülatörü"
    ])
    
    with tab1:
        st.subheader("🧪 Hipotez ve Varsayım Testleri")
        c_test, c_alpha = st.columns([3, 1])
        test_cat = c_test.selectbox(
            "Test Kategorisi", 
            [
                "Kendi Hipotezini Kur (Yönlü Test)",
                "İki Grup Karşılaştırma (T-Test & Mann-Whitney)", 
                "Çoklu Grup Karşılaştırma (ANOVA & Kruskal-Wallis)", 
                "Korelasyon Testleri (Pearson & Spearman)",
                "Normallik Testi (Shapiro-Wilk)"
            ]
        )
        alpha = c_alpha.selectbox("Anlamlılık Düzeyi (α)", [0.01, 0.05, 0.10], index=1)
        st.markdown("---")
        
        if test_cat == "Kendi Hipotezini Kur (Yönlü Test)":
            st.write("Belirlediğiniz iki değişken arasındaki neden-sonuç / yön ilişkisini (1-Tailed Test) test edin.")
            k1, k2, k3 = st.columns([2,2,3])
            v1 = k1.selectbox("Değişken (X):", num_cols, index=num_cols.index("3PA") if "3PA" in num_cols else 0)
            v2 = k2.selectbox("Değişken (Y):", num_cols, index=num_cols.index("WS") if "WS" in num_cols else 1)
            hipotez_yonu = k3.radio("Test Edilecek Hipoteziniz (H1):", [f"X ({v1}) arttıkça, Y ({v2}) ARTAR", f"X ({v1}) arttıkça, Y ({v2}) AZALIR"])
            
            if st.button("Hipotezi Test Et", use_container_width=True):
                df_clean = f_df.dropna(subset=[v1, v2])
                r_stat, p_val_2tailed = stats.pearsonr(df_clean[v1], df_clean[v2])
                p_val_1tailed = p_val_2tailed / 2
                
                st.write(f"**🔬 Sizin Hipoteziniz (H1):** {hipotez_yonu}")
                st.write(f"**⚪ Sıfır Hipotezi (H0):** {v1} ile {v2} arasında sizin belirttiğiniz yönde bir ilişki yoktur.")
                
                c_res1, c_res2 = st.columns(2)
                c_res1.metric("Korelasyon Katsayısı (r)", f"{r_stat:.3f}")
                c_res2.metric("Tek Yönlü P-Değeri (1-Tailed)", format_p(p_val_1tailed))
                
                beklenen_yon = 1 if "ARTAR" in hipotez_yonu else -1
                gercek_yon = 1 if r_stat > 0 else -1
                
                if p_val_1tailed < alpha and beklenen_yon == gercek_yon:
                    st.success(f"✅ **Hipoteziniz KABUL EDİLDİ!** İstatistiksel olarak p < {alpha} ve ilişki yönü eşleşiyor. Gerçekten de {hipotez_yonu}.")
                elif p_val_1tailed < alpha and beklenen_yon != gercek_yon:
                    st.error(f"🚨 **Hipoteziniz REDDEDİLDİ (Tam Ters Etki)!** İstatistiksel olarak güçlü bir ilişki var, ANCAK tam tersi yönde! (r = {r_stat:.3f})")
                else:
                    st.warning(f"❌ **Hipoteziniz REDDEDİLDİ.** Veriler hipotezinizi istatistiksel olarak destekleyecek kanıt sunmuyor.")

        elif test_cat == "İki Grup Karşılaştırma (T-Test & Mann-Whitney)":
            test_type = st.radio("Test Tipi:", ["Bağımsız Örneklem T-Testi (Parametrik)", "Mann-Whitney U Testi (Non-Parametrik)"], horizontal=True)
            if "T-Test" in test_type: st.latex(r"t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}")
            else: st.latex(r"U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1")
            t1, t2 = st.columns(2)
            hedef_var = t1.selectbox("Hedef (Test Edilecek) Değişken", num_cols, index=num_cols.index("Current_Salary") if "Current_Salary" in num_cols else 0)
            kriter_var = t2.selectbox("Sayısal Ayırıcı Kriter", num_cols, index=num_cols.index("3PA") if "3PA" in num_cols else 1)
            esik = st.slider(f"'{kriter_var}' Ayırma Eşiği", float(f_df[kriter_var].min()), float(f_df[kriter_var].max()), float(f_df[kriter_var].mean()))
            if st.button("Testi Çalıştır", use_container_width=True):
                g1 = f_df[f_df[kriter_var] > esik][hedef_var].dropna()
                g2 = f_df[f_df[kriter_var] <= esik][hedef_var].dropna()
                if len(g1) > 5 and len(g2) > 5:
                    if "T-Test" in test_type:
                        stat_val, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                        stat_name = "T-İstatistiği"
                    else:
                        stat_val, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                        stat_name = "U-İstatistiği"
                    c_res1, c_res2 = st.columns(2)
                    c_res1.metric(stat_name, f"{stat_val:.3f}")
                    c_res2.metric("P-Değeri (Anlamlılık)", format_p(p_val))
                    if p_val < alpha: st.success(f"✅ **Sonuç:** p < {alpha}. Gruplar arasında istatistiksel olarak **ANLAMLI** bir fark vardır.")
                    else: st.warning(f"❌ **Sonuç:** p >= {alpha}. İstatistiksel olarak anlamlı bir fark kanıtlanamamıştır.")
                else: st.error("Yetersiz veri.")

        elif test_cat == "Çoklu Grup Karşılaştırma (ANOVA & Kruskal-Wallis)":
            test_type = st.radio("Test Tipi:", ["Tek Yönlü ANOVA (Parametrik)", "Kruskal-Wallis (Non-Parametrik)"], horizontal=True)
            if "ANOVA" in test_type: st.latex(r"F = \frac{MS_{between}}{MS_{within}}")
            else: st.latex(r"H = \frac{12}{N(N+1)} \sum \frac{R_i^2}{n_i} - 3(N+1)")
            a1, a2 = st.columns(2)
            cat_var = a1.selectbox("Kategorik Değişken (Gruplar)", cat_cols)
            num_var = a2.selectbox("Hedef Sayısal Değişken", num_cols)
            if st.button("Testi Çalıştır", use_container_width=True):
                df_clean = f_df.dropna(subset=[cat_var, num_var])
                gruplar = [grup[num_var].values for isim, grup in df_clean.groupby(cat_var) if len(grup) > 5]
                if len(gruplar) > 1:
                    if "ANOVA" in test_type:
                        stat_val, p_val = stats.f_oneway(*gruplar)
                        stat_name = "F-İstatistiği"
                    else:
                        stat_val, p_val = stats.kruskal(*gruplar)
                        stat_name = "H-İstatistiği"
                    c_res1, c_res2 = st.columns(2)
                    c_res1.metric(stat_name, f"{stat_val:.3f}")
                    c_res2.metric("P-Değeri (Anlamlılık)", format_p(p_val))
                    if p_val < alpha: st.success(f"✅ **Sonuç:** p < {alpha}. Gruplar arasında istatistiksel olarak anlamlı farklar vardır.")
                    else: st.warning("❌ **Sonuç:** Anlamlı bir fark yoktur.")
                else: st.error("Yetersiz grup verisi.")

        elif test_cat == "Korelasyon Testleri (Pearson & Spearman)":
            test_type = st.radio("Test Tipi:", ["Pearson (Doğrusal)", "Spearman (Sıralı / Non-Parametrik)"], horizontal=True)
            if "Pearson" in test_type: st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}")
            else: st.latex(r"\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}")
            k1, k2 = st.columns(2)
            v1 = k1.selectbox("Değişken 1", num_cols, index=0)
            v2 = k2.selectbox("Değişken 2", num_cols, index=1)
            if st.button("Testi Çalıştır", use_container_width=True):
                df_clean = f_df.dropna(subset=[v1, v2])
                if "Pearson" in test_type:
                    stat_val, p_val = stats.pearsonr(df_clean[v1], df_clean[v2])
                    stat_name = "Pearson r"
                else:
                    stat_val, p_val = stats.spearmanr(df_clean[v1], df_clean[v2])
                    stat_name = "Spearman rho"
                c_res1, c_res2 = st.columns(2)
                c_res1.metric(stat_name, f"{stat_val:.3f}")
                c_res2.metric("P-Değeri", format_p(p_val))
                if p_val < alpha:
                    yon = "Pozitif" if stat_val > 0 else "Negatif"
                    st.success(f"✅ **Sonuç:** p < {alpha}. İki değişken arasında istatistiksel olarak anlamlı, **{yon}** yönlü bir ilişki vardır.")
                else: st.warning("❌ **Sonuç:** Anlamlı bir ilişki bulunamamıştır.")

        elif test_cat == "Normallik Testi (Shapiro-Wilk)":
            st.latex(r"W = \frac{(\sum a_i x_{(i)})^2}{\sum (x_i - \bar{x})^2}")
            n_var = st.selectbox("Test Edilecek Değişken", num_cols)
            if st.button("Testi Çalıştır", use_container_width=True):
                df_clean = f_df.dropna(subset=[n_var])
                if len(df_clean) > 3:
                    stat_val, p_val = stats.shapiro(df_clean[n_var])
                    c_res1, c_res2 = st.columns(2)
                    c_res1.metric("W-İstatistiği", f"{stat_val:.3f}")
                    c_res2.metric("P-Değeri", format_p(p_val))
                    if p_val < alpha: st.warning(f"❌ **Sonuç:** p < {alpha}. '{n_var}' Normal Dağılıma UYMAMAKTADIR.")
                    else: st.success(f"✅ **Sonuç:** p >= {alpha}. '{n_var}' Normal Dağılmaktadır.")

    with tab2:
        st.subheader("📊 Gelişmiş Regresyon: Adil Piyasa Değeri Tahmini")
        st.markdown("""
        Bu model, oyuncuların verimlilik ve savunma metriklerini kullanarak piyasa değerlerini hesaplar. 
        Özellikle **Modern Basketbol** metrikleri ($TS\%$, $eFG\%$, $3PAr$) ağırlıklı olarak dikkate alınır.
        """)
        
        # --- 1. VERİ HAZIRLIĞI ---
        current_season_df = f_df[f_df["Season"] == latest_season].copy()
        
        # AST/TO ve Verimlilik Kontrolleri
        if "AST" in current_season_df.columns and "TOV" in current_season_df.columns:
            current_season_df["AST/TO"] = current_season_df["AST"] / (current_season_df["TOV"] + 0.1)
        
        # Senin istediğin "Modern Basketbolun Kalbi" listesi
        default_features = ["Age", "TS%", "eFG%", "3PAr", "USG%", "AST/TO", "DWS", "DBPM", "VORP"]
        
        # Kullanılabilir sayısal sütunları belirle
        valid_num_cols = [col for col in num_cols if col not in ["Salary", "Current_Salary", "Season", "Player"]]
        for extra in ["AST/TO"]:
            if extra in current_season_df.columns and extra not in valid_num_cols:
                valid_num_cols.append(extra)
        
        # --- 2. MODEL PARAMETRELERİ ---
        c_mod1, c_mod2 = st.columns(2)
        model_type = c_mod1.radio("Algoritma Seçimi:", ["Ridge Regresyonu (L2 - Kararlı)", "Klasik Regresyon (OLS)"])
        use_position = c_mod2.checkbox("Pozisyon Etkisini Dahil Et", value=True, help="Pivot ve Guard maaş farklarını modele öğretir.")
        
        selected_features = st.multiselect(
            "Modele Dahil Edilecek Metrikler:", 
            valid_num_cols, 
            default=[f for f in default_features if f in valid_num_cols]
        )
        
        if len(selected_features) > 0:
            # --- 3. DİNAMİK LATEX FORMÜLÜ (Hatasız Versiyon) ---
            clean_f = [f.replace('%', r'\%') for f in selected_features]
            if len(clean_f) > 5:
                terms = [r"\beta_{" + str(i+1) + r"}(\text{" + f + r"})" for i, f in enumerate(clean_f[:4])]
                formula_str = r"\text{Salary} = \beta_0 + " + " + ".join(terms) + r" + \dots + \beta_n(\text{" + clean_f[-1] + r"}) + \epsilon"
            else:
                terms = [r"\beta_{" + str(i+1) + r"}(\text{" + f + r"})" for i, f in enumerate(clean_f)]
                formula_str = r"\text{Salary} = \beta_0 + " + " + ".join(terms) + r" + \epsilon"
            st.latex(formula_str)
            
            # --- 4. MODEL EĞİTİMİ ---
            cols_to_use = selected_features + ["Current_Salary"]
            if use_position and pos_col: cols_to_use.append(pos_col)
            
            reg_data = current_season_df[current_season_df["Current_Salary"] > 0].dropna(subset=cols_to_use).copy()
            
            if len(reg_data) > 20:
                X = reg_data[selected_features].copy()
                if use_position and pos_col:
                    pos_dummies = pd.get_dummies(reg_data[pos_col], prefix='Pos', drop_first=True).astype(int)
                    X = pd.concat([X, pos_dummies], axis=1)
                
                y = reg_data["Current_Salary"]
                
                # Model Fit
                model = Ridge(alpha=10.0).fit(X, y) if "Ridge" in model_type else LinearRegression().fit(X, y)
                
                reg_data["Expected_Salary"] = model.predict(X)
                reg_data["Difference"] = reg_data["Expected_Salary"] - reg_data["Current_Salary"]
                
                # --- 5. SONUÇLAR VE ANALİZ ---
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.write("🎯 **Tekil Oyuncu Analizi**")
                    target_player = st.selectbox("Oyuncu Seçin:", sorted(reg_data["Player"].unique()))
                    p_info = reg_data[reg_data["Player"] == target_player].iloc[0]
                    
                    st.metric("Gerçek Maaş", fmt_money(p_info['Current_Salary']))
                    st.metric("Tahmini (Adil) Değer", fmt_money(p_info['Expected_Salary']))
                    
                    diff = p_info['Difference']
                    color = "normal" if diff > 0 else "inverse"
                    st.metric("Durum", "Kelepir" if diff > 0 else "Aşırı Ödeme", delta=fmt_money(diff), delta_color=color)

                with col_res2:
                    st.write("📈 **Modelin 'Para Ödediği' Metrikler (Beta Katsayıları)**")
                    # Katsayıları tablo yapalım
                    coef_df = pd.DataFrame({
                        "Metrik": X.columns,
                        "Maaş Etkisi (Dolar)": [f"${int(c):,}" for c in model.coef_]
                    }).sort_values(by="Metrik")
                    st.dataframe(coef_df, hide_index=True, use_container_width=True)

                st.markdown("### 🏆 En Değerli / En Şişirilmiş Kontratlar")
                top_tab1, top_tab2 = st.tabs(["💎 Kelepir Oyuncular (Underpaid)", "💰 Fazla Maaş Alanlar (Overpaid)"])
                
                with top_tab1:
                    st.dataframe(reg_data.nlargest(10, "Difference")[["Player", "Current_Salary", "Expected_Salary", "Difference"]], use_container_width=True)
                with top_tab2:
                    st.dataframe(reg_data.nsmallest(10, "Difference")[["Player", "Current_Salary", "Expected_Salary", "Difference"]], use_container_width=True)
            else:
                st.warning("Seçilen metrikler için yeterli veri bulunamadı.")

    with tab3:
        st.subheader("🤖 K-Means: Pozisyonsuz Basketbol Rolleri")
        features = ["PTS", "TRB", "AST", "3PA", "BLK"]
        km_df = f_df[(f_df["Season"] == latest_season)].dropna(subset=features).copy()
        k_clusters = st.slider("Hedef Küme (Rol) Sayısı", 2, 8, 4)
        if len(km_df) > 10:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42)
            km_df.loc[:, "Cluster"] = kmeans.fit_predict(km_df[features])
            km_df["Cluster"] = km_df["Cluster"].astype(str)
            fig = px.scatter(km_df, x="PTS", y="TRB", color="Cluster", hover_data=["Player"], title="İstatistiksel Kümeleme")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🎲 Monte Carlo & AI Kadro Mühendisi")

        if 'ai_roster_key' not in st.session_state:
            st.session_state['ai_roster_key'] = []

        with st.expander("🧬 AI Kadro Kur (NBA Maaş Kuralları & Şampiyonluk Optimizasyonu)", expanded=True):
            mode = st.radio("Optimizasyon Hedefi:", ["Aggressive (Maksimum WS)", "Moneyball (Verimlilik)"])
            c_sa1, c_sa2 = st.columns(2)
            cap_m_sa = c_sa1.slider("Maaş Tavanı (Milyon $)", 100, 300, 140)
            roster_size = c_sa2.slider("Kadro Büyüklüğü (Kişi)", 12, 15, 15)
            cap_sa = cap_m_sa * 1000000
            
            if st.button("🧬 NBA Kurallarına Göre Kadro Kur", use_container_width=True):
                if pos_col:
                    # Sütunları dinamik bul
                    c_ws = next((x for x in ["WS", "WS_adv"] if x in f_df.columns), "WS")
                    c_bpm = next((x for x in ["BPM", "BPM_adv"] if x in f_df.columns), "BPM")
                    c_usg = next((x for x in ["USG%", "USG%_per"] if x in f_df.columns), "USG%")
                    c_g = next((x for x in ["G", "G_per"] if x in f_df.columns), "G")

                    sa_df = f_df.groupby('Player').agg({
                        c_ws: 'mean', c_bpm: 'mean', c_usg: 'mean', c_g: 'mean', pos_col: 'first'
                    }).reset_index().fillna(0).rename(columns={pos_col: 'Pos', c_ws: 'WS', c_bpm: 'BPM', c_usg: 'USG%', c_g: 'G'})
                    
                    # --- NBA MAAŞ KURALLARI ENTEGRASYONU ---
                    sa_df['Final_Salary'] = sa_df['Player'].map(latest_salaries).fillna(1100000) # Min Maaş: 1.1M$
                    
                    # 1. KURAL: Maksimum Kontrat Sınırı (Bir oyuncu cap'in %35'ini geçemez)
                    max_contract_limit = cap_sa * 0.35
                    sa_df = sa_df[sa_df['Final_Salary'] <= max_contract_limit]
                    
                    if len(sa_df) >= roster_size:
                        with st.spinner("AI şampiyonluk kadrosunu CBA kurallarına göre dizayn ediyor..."):
                            players = sa_df['Player'].tolist()
                            prob = pulp.LpProblem("NBA_CBA_Opt", pulp.LpMaximize)
                            p_vars = pulp.LpVariable.dicts("P", players, 0, 1, pulp.LpBinary)
                            
                            # Hedef Fonksiyon
                            if mode == "Aggressive (Maksimum WS)":
                                prob += pulp.lpSum([(sa_df.loc[sa_df['Player']==p, 'WS'].values[0] + sa_df.loc[sa_df['Player']==p, 'BPM'].values[0]*0.5) * p_vars[p] for p in players])
                            else:
                                prob += pulp.lpSum([(sa_df.loc[sa_df['Player']==p, 'WS'].values[0] / (sa_df.loc[sa_df['Player']==p, 'Final_Salary'].values[0]/1000000 + 1)) * p_vars[p] for p in players])
                            
                            # --- KISITLAR ---
                            # 2. KURAL: Salary Cap (Üst Sınır)
                            prob += pulp.lpSum([sa_df.loc[sa_df['Player']==p, 'Final_Salary'].values[0] * p_vars[p] for p in players]) <= cap_sa
                            
                            # 3. KURAL: Salary Floor (Alt Sınır - Bütçenin en az %85'ini harcamalı)
                            prob += pulp.lpSum([sa_df.loc[sa_df['Player']==p, 'Final_Salary'].values[0] * p_vars[p] for p in players]) >= (cap_sa * 0.85)
                            
                            prob += pulp.lpSum([p_vars[p] for p in players]) == roster_size
                            
                            # Pozisyon ve Yıldız Kısıtları
                            guards = [p for p in players if any(x in str(sa_df.loc[sa_df['Player']==p, 'Pos'].values[0]).upper() for x in ["G", "PG", "SG"])]
                            forwards = [p for p in players if any(x in str(sa_df.loc[sa_df['Player']==p, 'Pos'].values[0]).upper() for x in ["F", "SF", "PF"])]
                            centers = [p for p in players if "C" in str(sa_df.loc[sa_df['Player']==p, 'Pos'].values[0]).upper()]
                            prob += pulp.lpSum([p_vars[p] for p in guards]) >= 4
                            prob += pulp.lpSum([p_vars[p] for p in forwards]) >= 4
                            prob += pulp.lpSum([p_vars[p] for p in centers]) >= 2
                            
                            stars = [p for p in players if sa_df.loc[sa_df['Player']==p, 'USG%'].values[0] >= 25.0]
                            prob += pulp.lpSum([p_vars[p] for p in stars]) <= 2
                            
                            prob.solve(pulp.PULP_CBC_CMD(msg=0))
                            
                            if pulp.LpStatus[prob.status] == 'Optimal':
                                st.session_state['ai_roster_key'] = [p for p in players if p_vars[p].varValue == 1.0]
                                st.rerun()
                            else:
                                st.error("❌ CBA Kurallarına uygun kadro bulunamadı. Maaş tavanını veya oyuncu sayısını değiştirin.")

        st.markdown("---")
        sim_roster = st.multiselect("Seçili Kadro:", options=available_players, key='ai_roster_key')
        
        if len(sim_roster) >= 5:
            # 1. VERİLERİ ÇEK VE HAZIRLA
            c_ws = next((x for x in ["WS", "WS_adv"] if x in f_df.columns), "WS")
            c_bpm = next((x for x in ["BPM", "BPM_adv"] if x in f_df.columns), "BPM")
            c_mp = next((x for x in ["MP", "MP_per"] if x in f_df.columns), "MP")
            c_g = next((x for x in ["G", "G_per"] if x in f_df.columns), "G")

            m_df = f_df[f_df["Player"].isin(sim_roster)].groupby('Player').agg({
                c_ws: 'mean', c_bpm: 'mean', c_mp: 'mean', c_g: 'mean', pos_col: 'first'
            }).reset_index().fillna(0).rename(columns={c_ws:'WS', c_bpm:'BPM', c_mp:'MP', c_g:'G', pos_col:'Pos'})
            
            # --- GERÇEKÇİ ROTASYON HESABI (84 WS'yi DÜZELTEN KISIM) ---
            # Sahada toplam 240 dakika var. Oyuncuların sürelerini bu sınıra göre normalize ediyoruz.
            total_planned_mp = m_df['MP'].sum()
            normalization_factor = 240 / total_planned_mp if total_planned_mp > 240 else 1.0
            
            # Oyuncunun yeni WS'si = (Eski WS / Oynadığı Maç) * (Normalize Edilmiş Dakika) * (Oynayabileceği Maç)
            # Daha basitçe: Toplam WS'yi 240 dakikaya oranlıyoruz.
            m_df['Realistic_WS'] = m_df['WS'] * normalization_factor
            
            # Sinerji ve Ceza (USG% yine önemli)
            total_usg = f_df[f_df["Player"].isin(sim_roster)].groupby('Player')['USG%'].mean().nlargest(5).sum()
            usage_penalty = max(0.85, 1.0 - ((total_usg - 115) * 0.005)) if total_usg > 115 else 1.0
            
            # 10.000 Sezonluk Monte Carlo
            sim_results = []
            for _ in range(10000):
                # Her sezon oyuncuların sakatlık durumuna göre WS üretimi
                season_ws = sum([row['Realistic_WS'] * (np.random.binomial(82, min(row['G']/82.0, 0.95))/82.0) for _, row in m_df.iterrows()])
                # Şans faktörü ve Sinerji ekle
                final_wins = np.random.normal(season_ws * usage_penalty, 3.5)
                # NBA'de bir takım 82'den fazla, 0'dan az galibiyet alamaz
                sim_results.append(max(0, min(82, final_wins)))
            
            sims = np.array(sim_results)
            avg_wins = sims.mean()
            
            # --- EKRAN ÇIKTILARI (ARTIK GERÇEKÇİ) ---
            c1, c2, c3 = st.columns(3)
            # Ham toplam yerine 'Tahmini Galibiyet' olarak gösterelim
            c1.metric("Tahmini Sezon Sonu Galibiyet", f"{avg_wins:.1f}")
            c2.metric("Sinerji Etkisi", f"x{usage_penalty:.2f}")
            c3.metric("Playoff Şansı", f"%{(len(sims[sims >= 44])/10000)*100:.1f}")
            
            st.write(f"ℹ️ **Not:** Toplam WS ({m_df['WS'].sum():.1f}), NBA rotasyon kurallarına (maç başı 240 dk) göre **{avg_wins:.1f}** seviyesine normalize edilmiştir.")
            
            st.plotly_chart(px.histogram(sims, title="Galibiyet Dağılımı (Rotasyon Ayarlı)"), use_container_width=True)
            st.dataframe(m_df[['Player', 'Pos', 'WS', 'Realistic_WS', 'MP']], use_container_width=True)