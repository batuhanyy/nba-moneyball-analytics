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

# 📌 HATA BURADA ÇÖZÜLDÜ: Session State, "available_players" tanımlandıktan sonra çalıştırıldı.
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
        st.subheader("📊 Çoklu Doğrusal Regresyon: Adil Piyasa Değeri Tahmini")
        st.latex(r"\text{Expected Salary} = \beta_0 + \beta_1(\text{PTS}) + \beta_2(\text{AST}) + \beta_3(\text{TRB}) + \beta_4(\text{WS}) + \epsilon")
        reg_df = f_df[(f_df["Season"] == latest_season) & (f_df["Current_Salary"] > 0)].dropna(subset=["PTS", "AST", "TRB", "WS", "Current_Salary"])
        if len(reg_df) > 10:
            X = reg_df[["PTS", "AST", "TRB", "WS"]]
            y = reg_df["Current_Salary"]
            mlr_model = LinearRegression().fit(X, y)
            reg_df["Expected_Salary"] = mlr_model.predict(X)
            reg_df["Difference"] = reg_df["Expected_Salary"] - reg_df["Current_Salary"]
            p_sel = st.selectbox("Model Üzerinde Test Edilecek Oyuncu:", sorted(reg_df["Player"].unique()))
            p_res = reg_df[reg_df["Player"] == p_sel].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Gerçek Maaşı", fmt_money(p_res['Current_Salary']))
            c2.metric("Adil Maaş", fmt_money(p_res['Expected_Salary']))
            if p_res['Difference'] > 0: c3.metric("Durum", "Ucuz", delta=f"+{fmt_money(p_res['Difference'])} Kar")
            else: c3.metric("Durum", "Pahalı", delta=f"-{fmt_money(abs(p_res['Difference']))} Zarar", delta_color="red")

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
        st.write("Kendi belirlediğiniz oyuncularla simülasyon yapabilir veya Yapay Zekanın optimum kadroyu sizin yerinize bulup simülasyona aktarmasını sağlayabilirsiniz.")
        
        with st.expander("🧬 Yapay Zeka ile Otomatik Kadro Kur (Simulated Annealing)", expanded=False):
            st.markdown("Bütçenize ve pozisyon kurallarına (En az 2 PG, SG, vb.) uyan optimum 13 kişilik kadroyu hesaplar.")
            cap_m_sa = st.slider("Maaş Tavanı (Milyon $)", 100, 250, 140, key="cap_sa")
            cap_sa = cap_m_sa * 1000000
            
            if st.button("🧬 Kadroyu Kur ve Simülasyona Aktar", use_container_width=True):
                if pos_col:
                    sa_df = f_df.groupby('Player').agg({'WS': 'mean', 'Current_Salary': 'first', pos_col: 'first'}).reset_index()
                    sa_df = sa_df.rename(columns={pos_col: 'Pos'})
                    sa_df['Final_Salary'] = sa_df['Player'].map(latest_salaries)
                    sa_df = sa_df[(sa_df['Final_Salary'] > 0) & (sa_df['Final_Salary'] <= cap_sa * 0.35)].dropna(subset=['WS'])
                    
                    if len(sa_df) > 15:
                        with st.spinner("Yapay Zeka on binlerce kombinasyonu tarıyor, lütfen bekleyin..."):
                            current_roster = list(sa_df.sample(13)['Player'].values)
                            
                            def eval_roster(roster_names):
                                r_df = sa_df[sa_df['Player'].isin(roster_names)]
                                cost = r_df['Final_Salary'].sum()
                                if cost > cap_sa or cost < cap_sa * 0.9: return -999 
                                
                                p_str = r_df['Pos'].astype(str).str.upper()
                                g_c = sum(p_str.str.contains('PG|SG'))
                                w_c = sum(p_str.str.contains('SF'))
                                b_c = sum(p_str.str.contains('PF|C'))
                                if g_c < 4 or w_c < 2 or b_c < 3: return -500 
                                
                                return r_df['WS'].sum()
                            
                            current_score = eval_roster(current_roster)
                            best_roster = current_roster.copy()
                            best_score = current_score
                            T = 1000.0
                            cooling_rate = 0.95
                            all_players = sa_df['Player'].values
                            
                            for i in range(1000):
                                new_roster = current_roster.copy()
                                p_out = np.random.choice(new_roster)
                                p_in = np.random.choice([p for p in all_players if p not in new_roster])
                                new_roster.remove(p_out)
                                new_roster.append(p_in)
                                new_score = eval_roster(new_roster)
                                
                                if new_score > current_score:
                                    current_roster = new_roster
                                    current_score = new_score
                                    if new_score > best_score:
                                        best_roster = new_roster.copy()
                                        best_score = new_score
                                else:
                                    acceptance_prob = math.exp((new_score - current_score) / T) if T > 0.1 else 0
                                    if np.random.rand() < acceptance_prob:
                                        current_roster = new_roster
                                        current_score = new_score
                                T *= cooling_rate
                            
                            if best_score > 0:
                                st.session_state['roster_selection'] = best_roster
                                st.success("✅ Optimum kadro bulundu ve aşağıdaki simülasyona aktarıldı!")
                            else:
                                st.error("Geçerli kadro bulunamadı. Bütçeyi artırın.")
                    else:
                        st.error("Yeterli oyuncu havuzu bulunamadı.")
                else:
                    st.error("Pozisyon sütunu bulunamadı.")
        
        st.markdown("---")
        
        sim_roster = st.multiselect("Simülasyon İçin Kadro Seçin", available_players, key="roster_selection")
        
        if len(sim_roster) >= 5: 
            agg_dict = {'WS': 'mean', 'BPM': 'mean', 'VORP': 'mean', 'USG%': 'mean', 'G': 'mean'}
            if pos_col: agg_dict[pos_col] = 'first'
                
            m_df = f_df[f_df["Player"].isin(sim_roster)].groupby('Player').agg(agg_dict).reset_index()
            m_df['Player_Value'] = m_df['WS'] + (m_df['BPM'] * 0.5) + m_df['VORP']
            
            total_usg = m_df.nlargest(5, 'USG%')['USG%'].sum()
            usage_penalty = 1.0
            if total_usg > 115.0: usage_penalty = max(0.75, 1.0 - ((total_usg - 115) * 0.005))
            
            synergy_mult = 1.0
            g_count, w_count, b_count = 0, 0, 0
            if pos_col:
                m_df['Pos_str'] = m_df[pos_col].astype(str).str.upper()
                g_count = sum(m_df['Pos_str'].str.contains('PG|SG'))
                w_count = sum(m_df['Pos_str'].str.contains('SF'))
                b_count = sum(m_df['Pos_str'].str.contains('PF|C'))
                if g_count < 2: synergy_mult -= 0.1
                if w_count < 1: synergy_mult -= 0.05
                if b_count < 2: synergy_mult -= 0.1
            
            sim_results = []
            for _ in range(10000):
                season_value = 0
                for idx, row in m_df.iterrows():
                    health_prob = min(row['G'] / 82.0, 0.95) if row['G'] > 0 else 0.85
                    games_played = np.random.binomial(82, health_prob)
                    season_value += row['Player_Value'] * (games_played / 82.0)
                
                team_wins = season_value * usage_penalty * synergy_mult
                team_wins = np.random.normal(team_wins, 3.5)
                sim_results.append(team_wins)
                
            simulations = np.array(sim_results)
            playoff_teams = simulations[simulations >= 45]
            playoff_prob = (len(playoff_teams) / 10000) * 100
            
            champ_count = 0
            for w in playoff_teams:
                rounds = [48, 52, 55, 58]
                won_championship = True
                for opp_w in rounds:
                    p_win_game = w / (w + opp_w)
                    p_win_series = sum([math.comb(7, k) * (p_win_game**k) * ((1-p_win_game)**(7-k)) for k in range(4, 8)])
                    if np.random.rand() > p_win_series:
                        won_championship = False
                        break
                if won_championship: champ_count += 1
            
            champ_prob = (champ_count / 10000) * 100
            
            c_ws, c_adv, c_syn = st.columns(3)
            c_ws.metric("Kadro Saf Yeteneği (WS Toplamı)", f"{m_df['WS'].sum():.1f}")
            c_adv.metric("Gelişmiş Güç İndeksi", f"{m_df['Player_Value'].sum():.1f}")
            c_syn.metric("Sinerji & Top Kullanım Çarpanı", f"x{(usage_penalty * synergy_mult):.2f}")
            
            c_p, c_c = st.columns(2)
            c_p.metric("Playoff'a Kalma İhtimali", f"%{playoff_prob:.1f}")
            c_c.metric("Şampiyonluk İhtimali (Bracket Sim)", f"%{champ_prob:.1f}")
            
            fig = px.histogram(simulations, nbins=50, title="10.000 Sezonluk Monte Carlo Dağılımı")
            fig.add_vline(x=45, line_dash="dash", line_color="yellow", annotation_text="Playoff Barajı")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Takım Analiz Raporunu Göster"):
                st.write(f"- **Top Kullanım Çatışması:** Kadrodaki top kullanım oranı %{total_usg:.1f}. (Ceza: %{(1-usage_penalty)*100:.1f})")
                st.write(f"- **Pozisyon Dağılımı:** {g_count} Guard, {w_count} Wing, {b_count} Big")
            
            st.markdown("---")
            st.subheader("📋 Simüle Edilen Takım Kadrosu Detayları")
            roster_table_data = []
            for p in sim_roster:
                p_stats = f_df[f_df["Player"] == p].sort_values("Season").iloc[-1]
                row_dict = {"Oyuncu": p, "Maaş": fmt_money(latest_salaries.get(p, 0)), "WS": round(p_stats.get("WS", 0), 2), "BPM": round(p_stats.get("BPM", 0), 2), "VORP": round(p_stats.get("VORP", 0), 2), "USG%": round(p_stats.get("USG%", 0), 1)}
                if pos_col: row_dict["Pozisyon"] = p_stats.get(pos_col, "-")
                roster_table_data.append(row_dict)
                
            roster_display_df = pd.DataFrame(roster_table_data)
            col_order = ["Oyuncu"] + (["Pozisyon"] if pos_col else []) + ["Maaş", "WS", "BPM", "VORP", "USG%"]
            st.dataframe(roster_display_df[col_order], use_container_width=True)

        else:
            st.warning("Simülasyonun çalışması için lütfen en az 5 oyuncu seçin.")